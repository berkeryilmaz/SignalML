import json
from typing import Dict, List, Any, Optional, Tuple, Union
from collections import defaultdict, deque
import numpy as np
import tensorflow as tf
import keras
from keras import layers, ops, models


class KerasGraphError(Exception):
    """Raised when Keras node graph structure or tensor shape is invalid."""
    pass


# Custom Positional Encoding Layer for 1D/Transformer Signals
@keras.saving.register_keras_serializable(package="signalml")
class PositionalEncoding(layers.Layer):
    def __init__(self, d_model: int, max_len: int = 5000, **kwargs):
        super().__init__(**kwargs)
        self.d_model = d_model
        self.max_len = max_len
        
        pe = np.zeros((max_len, d_model), dtype=np.float32)
        position = np.arange(0, max_len, dtype=np.float32)[:, np.newaxis]
        div_term = np.exp(np.arange(0, d_model, 2, dtype=np.float32) * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = np.sin(position * div_term)
        pe[:, 1::2] = np.cos(position * div_term)
        self.pe = tf.constant(pe[np.newaxis, ...], dtype=tf.float32)

    def call(self, inputs):
        seq_len = tf.shape(inputs)[1]
        return inputs + self.pe[:, :seq_len, :]

    def get_config(self):
        config = super().get_config()
        config.update({"d_model": self.d_model, "max_len": self.max_len})
        return config


class KerasModelBuilder:
    @staticmethod
    def parse_graph(graph_data: Dict[str, Any]) -> Tuple[Dict[str, Dict[str, Any]], List[Dict[str, Any]]]:
        """Parses and normalizes nodes and links from graph payload."""
        raw_nodes = graph_data.get("nodes", [])
        raw_links = graph_data.get("links", [])

        nodes_by_id = {}
        for n in raw_nodes:
            nid = str(n.get("id"))
            nodes_by_id[nid] = n

        links = []
        for l in raw_links:
            if isinstance(l, list) and len(l) >= 5:
                # LiteGraph link: [id, origin_id, origin_slot, target_id, target_slot, type]
                links.append({
                    "id": str(l[0]),
                    "origin_id": str(l[1]),
                    "origin_slot": int(l[2]),
                    "target_id": str(l[3]),
                    "target_slot": int(l[4]),
                })
            elif isinstance(l, dict):
                links.append({
                    "id": str(l.get("id", "")),
                    "origin_id": str(l.get("origin_id")),
                    "origin_slot": int(l.get("origin_slot", 0)),
                    "target_id": str(l.get("target_id")),
                    "target_slot": int(l.get("target_slot", 0)),
                })

        return nodes_by_id, links

    @staticmethod
    def clean_node_type(node_type: str) -> str:
        """Normalizes node type string by removing prefixes like 'keras/' and normalizing punctuation."""
        t = str(node_type or "").lower().strip()
        if "/" in t:
            t = t.split("/")[-1]
        return t.replace(" ", "").replace("_", "")

    @staticmethod
    def _create_keras_layer(node_type: str, props: Dict[str, Any], node_id: str) -> Any:
        """Instantiates the corresponding Keras layer based on node type and properties."""
        t = KerasModelBuilder.clean_node_type(node_type)
        name = f"{t}_{node_id}"

        # 1. Core Layers
        if t in ("input", "inputlayer"):
            shape_str = str(props.get("shape", props.get("input_shape", "512, 1")))
            shape = tuple(int(s.strip()) for s in shape_str.split(",") if s.strip())
            return layers.Input(shape=shape, name=name)

        elif t == "dense":
            units = int(props.get("units", 64))
            act = props.get("activation", "relu")
            return layers.Dense(units=units, activation=act, name=name)

        elif t == "activation":
            act = props.get("activation", "relu")
            return layers.Activation(act, name=name)

        elif t in ("output", "modeloutput"):
            # Explicit output termination marker for AE, U-Net, etc.
            act = props.get("activation", "linear")
            return layers.Activation(act, name=name)

        elif t == "flatten":
            return layers.Flatten(name=name)

        elif t == "reshape":
            shape_str = str(props.get("target_shape", "-1, 1"))
            target_shape = tuple(int(s.strip()) for s in shape_str.split(",") if s.strip())
            return layers.Reshape(target_shape, name=name)

        elif t == "dropout":
            rate = float(props.get("rate", 0.2))
            return layers.Dropout(rate=rate, name=name)

        elif t in ("batchnormalization", "batchnorm", "bn"):
            return layers.BatchNormalization(name=name)

        elif t in ("layernormalization", "layernorm", "ln"):
            return layers.LayerNormalization(name=name)

        elif t in ("unitnormalization", "unitnorm"):
            return layers.UnitNormalization(axis=-1, name=name)

        elif t == "embedding":
            input_dim = int(props.get("input_dim", 1000))
            output_dim = int(props.get("output_dim", 64))
            return layers.Embedding(input_dim=input_dim, output_dim=output_dim, name=name)

        # 2. Convolutional Layers
        elif t == "conv1d":
            filters = int(props.get("filters", 32))
            kernel_size = int(props.get("kernel_size", 5))
            strides = int(props.get("strides", 1))
            padding = props.get("padding", "same")
            act = props.get("activation", "relu")
            return layers.Conv1D(filters=filters, kernel_size=kernel_size, strides=strides, padding=padding, activation=act, name=name)

        elif t in ("conv1dtranspose", "deconv1d"):
            filters = int(props.get("filters", 32))
            kernel_size = int(props.get("kernel_size", 5))
            strides = int(props.get("strides", 2))
            padding = props.get("padding", "same")
            act = props.get("activation", "relu")
            return layers.Conv1DTranspose(filters=filters, kernel_size=kernel_size, strides=strides, padding=padding, activation=act, name=name)

        elif t == "separableconv1d":
            filters = int(props.get("filters", 32))
            kernel_size = int(props.get("kernel_size", 5))
            padding = props.get("padding", "same")
            act = props.get("activation", "relu")
            return layers.SeparableConv1D(filters=filters, kernel_size=kernel_size, padding=padding, activation=act, name=name)

        elif t == "conv2d":
            filters = int(props.get("filters", 32))
            kernel_size = int(props.get("kernel_size", 3))
            padding = props.get("padding", "same")
            act = props.get("activation", "relu")
            return layers.Conv2D(filters=filters, kernel_size=kernel_size, padding=padding, activation=act, name=name)

        elif t in ("maxpooling1d", "maxpool1d"):
            pool_size = int(props.get("pool_size", 2))
            padding = props.get("padding", "same")
            return layers.MaxPooling1D(pool_size=pool_size, padding=padding, name=name)

        elif t in ("averagepooling1d", "avgpool1d"):
            pool_size = int(props.get("pool_size", 2))
            padding = props.get("padding", "same")
            return layers.AveragePooling1D(pool_size=pool_size, padding=padding, name=name)

        elif t in ("maxpooling2d", "maxpool2d"):
            pool_size = int(props.get("pool_size", 2))
            return layers.MaxPooling2D(pool_size=pool_size, name=name)

        elif t in ("averagepooling2d", "avgpool2d"):
            pool_size = int(props.get("pool_size", 2))
            return layers.AveragePooling2D(pool_size=pool_size, name=name)

        elif t in ("globalmaxpooling1d", "globalmaxpool1d"):
            return layers.GlobalMaxPooling1D(name=name)

        elif t in ("globalaveragepooling1d", "globalavgpool1d"):
            return layers.GlobalAveragePooling1D(name=name)

        elif t in ("globalmaxpooling2d", "globalmaxpool2d"):
            return layers.GlobalMaxPooling2D(name=name)

        elif t in ("globalaveragepooling2d", "globalavgpool2d"):
            return layers.GlobalAveragePooling2D(name=name)

        elif t in ("zeropadding1d", "zeropadding"):
            pad = int(props.get("padding", 1))
            return layers.ZeroPadding1D(padding=pad, name=name)

        elif t == "zeropadding2d":
            pad = int(props.get("padding", 1))
            return layers.ZeroPadding2D(padding=pad, name=name)

        elif t == "upsampling1d":
            size = int(props.get("size", 2))
            return layers.UpSampling1D(size=size, name=name)

        elif t == "upsampling2d":
            size = int(props.get("size", 2))
            return layers.UpSampling2D(size=size, name=name)

        # 3. Recurrent Layers
        elif t == "simplernn":
            units = int(props.get("units", 64))
            ret_seq = bool(props.get("return_sequences", False))
            act = props.get("activation", "tanh")
            return layers.SimpleRNN(units=units, return_sequences=ret_seq, activation=act, name=name)

        elif t == "lstm":
            units = int(props.get("units", 64))
            ret_seq = bool(props.get("return_sequences", False))
            drop = float(props.get("dropout", 0.0))
            rec_drop = float(props.get("recurrent_dropout", 0.0))
            return layers.LSTM(units=units, return_sequences=ret_seq, dropout=drop, recurrent_dropout=rec_drop, name=name)

        elif t == "gru":
            units = int(props.get("units", 64))
            ret_seq = bool(props.get("return_sequences", False))
            return layers.GRU(units=units, return_sequences=ret_seq, name=name)

        elif t == "bidirectional":
            sub_type = props.get("recurrent_type", "lstm").lower()
            units = int(props.get("units", 64))
            ret_seq = bool(props.get("return_sequences", False))
            if sub_type == "gru":
                rec_layer = layers.GRU(units, return_sequences=ret_seq)
            else:
                rec_layer = layers.LSTM(units, return_sequences=ret_seq)
            return layers.Bidirectional(rec_layer, name=name)

        elif t == "timedistributed":
            units = int(props.get("units", 64))
            act = props.get("activation", "relu")
            return layers.TimeDistributed(layers.Dense(units, activation=act), name=name)

        # 4. Attention & Transformer Layers
        elif t == "multiheadattention":
            num_heads = int(props.get("num_heads", 4))
            key_dim = int(props.get("key_dim", 32))
            dropout = float(props.get("dropout", 0.0))
            return layers.MultiHeadAttention(num_heads=num_heads, key_dim=key_dim, dropout=dropout, name=name)

        elif t == "attention":
            return layers.Attention(name=name)

        elif t == "positionalencoding":
            d_model = int(props.get("d_model", 64))
            max_len = int(props.get("max_len", 2048))
            return PositionalEncoding(d_model=d_model, max_len=max_len, name=name)

        # 5. Tensor Merge & Combinations
        elif t == "concatenate":
            axis = int(props.get("axis", -1))
            return layers.Concatenate(axis=axis, name=name)

        elif t == "add":
            return layers.Add(name=name)

        elif t == "multiply":
            return layers.Multiply(name=name)

        elif t == "average":
            return layers.Average(name=name)

        elif t == "dot":
            axes = int(props.get("axes", -1))
            return layers.Dot(axes=axes, name=name)

        elif t == "identity":
            return layers.Identity(name=name)

        else:
            raise KerasGraphError(f"Unsupported layer type '{node_type}' in node {node_id}")

    @classmethod
    def build_model(cls, graph_data: Dict[str, Any]) -> Tuple[keras.Model, Dict[str, Any]]:
        """
        Constructs a genuine Keras Functional API model from visual graph.
        Checks cycles, builds topological order, connects layer tensors, and verifies output shapes.
        """
        nodes_by_id, links = cls.parse_graph(graph_data)

        if not nodes_by_id:
            raise KerasGraphError("Graph has no nodes.")

        # Build in/out degree and incoming links map
        in_degree = {nid: 0 for nid in nodes_by_id}
        adj = defaultdict(list)
        # incoming_links: target_node_id -> list of links ordered by target_slot
        incoming_links = defaultdict(list)

        for l in links:
            src = l["origin_id"]
            dst = l["target_id"]
            if src in nodes_by_id and dst in nodes_by_id:
                adj[src].append(dst)
                in_degree[dst] += 1
                incoming_links[dst].append(l)

        # Sort incoming links by target_slot to preserve multi-input order
        for dst in incoming_links:
            incoming_links[dst].sort(key=lambda x: x["target_slot"])

        # Cycle detection & Topological Sort
        queue = deque([nid for nid, deg in in_degree.items() if deg == 0])
        topo_order = []

        while queue:
            curr = queue.popleft()
            topo_order.append(curr)
            for nxt in adj[curr]:
                in_degree[nxt] -= 1
                if in_degree[nxt] == 0:
                    queue.append(nxt)

        if len(topo_order) < len(nodes_by_id):
            raise KerasGraphError("Cycle detected in Neural Network graph! Keras requires an acyclic graph.")

        # Identify input nodes (supports both 'input' and prefixed 'keras/input')
        input_nodes = [
            nid for nid in topo_order
            if KerasModelBuilder.clean_node_type(nodes_by_id[nid].get("type", "")) in ("input", "inputlayer")
        ]
        if not input_nodes:
            raise KerasGraphError("Neural Network graph must have at least one Input node.")

        tensor_outputs: Dict[str, Any] = {}
        layer_summary: List[Dict[str, Any]] = []

        # Construct Functional model layer by layer
        for nid in topo_order:
            node = nodes_by_id[nid]
            ntype = node.get("type", "")
            clean_ntype = KerasModelBuilder.clean_node_type(ntype)
            props = node.get("properties", {})
            # Also read from widgets_values if properties are empty
            w_vals = node.get("widgets_values")
            if isinstance(w_vals, list) and len(w_vals) > 0 and not props:
                props = {"units": w_vals[0]}

            # Case A: Input node
            if clean_ntype in ("input", "inputlayer"):
                try:
                    tensor = cls._create_keras_layer(ntype, props, nid)
                    tensor_outputs[nid] = tensor
                    layer_summary.append({
                        "node_id": nid,
                        "layer_type": "Input",
                        "output_shape": list(tensor.shape),
                        "params": 0
                    })
                except Exception as e:
                    raise KerasGraphError(f"Error configuring Input node {nid}: {str(e)}")
                continue

            # Case B: Intermediate or Output node
            in_links = incoming_links.get(nid, [])
            if not in_links:
                raise KerasGraphError(f"Layer node {node.get('title', ntype)} (id: {nid}) has no input connection.")

            # Gather input tensors
            parent_tensors = []
            for link in in_links:
                origin_id = link["origin_id"]
                if origin_id not in tensor_outputs:
                    raise KerasGraphError(f"Missing predecessor tensor from node {origin_id} for node {nid}")
                parent_tensors.append(tensor_outputs[origin_id])

            try:
                layer_obj = cls._create_keras_layer(ntype, props, nid)

                # Connect layer
                if clean_ntype in ("concatenate", "add", "multiply", "average"):
                    if len(parent_tensors) < 2:
                        # If only one input connected to merge layer, pass it directly
                        out_tensor = parent_tensors[0]
                    else:
                        out_tensor = layer_obj(parent_tensors)
                elif clean_ntype in ("multiheadattention", "mha"):
                    # MultiHeadAttention query, key, value
                    if len(parent_tensors) == 1:
                        # Self-attention: query=tensor, value=tensor
                        out_tensor = layer_obj(query=parent_tensors[0], value=parent_tensors[0])
                    else:
                        out_tensor = layer_obj(query=parent_tensors[0], value=parent_tensors[1])
                else:
                    # Single input layer
                    input_tensor = parent_tensors[0]
                    # Check auto-channel expansion if Conv1D or LSTM gets 2D input (batch, seq)
                    if clean_ntype in ("conv1d", "separableconv1d", "conv1dtranspose", "lstm", "gru", "simplernn", "bidirectional"):
                        if len(input_tensor.shape) == 2:
                            input_tensor = ops.expand_dims(input_tensor, axis=-1)
                    
                    out_tensor = layer_obj(input_tensor)

                tensor_outputs[nid] = out_tensor

                # Extract shape and parameter count
                out_shape = [dim if dim is not None else -1 for dim in list(out_tensor.shape)]
                layer_summary.append({
                    "node_id": nid,
                    "layer_type": ntype,
                    "output_shape": out_shape,
                    "params": layer_obj.count_params() if hasattr(layer_obj, "count_params") else 0
                })

            except Exception as e:
                raise KerasGraphError(f"Failed to build layer '{ntype}' (id: {nid}): {str(e)}")

        # Find output nodes: if explicit 'output' nodes exist, prioritize them; else use terminal leaf nodes
        explicit_outputs = [
            nid for nid in topo_order
            if KerasModelBuilder.clean_node_type(nodes_by_id[nid].get("type", "")) in ("output", "modeloutput")
        ]
        if explicit_outputs:
            output_nodes = explicit_outputs
        else:
            output_nodes = [nid for nid in topo_order if len(adj[nid]) == 0]
            if not output_nodes:
                output_nodes = [topo_order[-1]]

        model_inputs = [tensor_outputs[nid] for nid in input_nodes]
        model_outputs = [tensor_outputs[nid] for nid in output_nodes]

        if len(model_inputs) == 1:
            model_inputs = model_inputs[0]
        if len(model_outputs) == 1:
            model_outputs = model_outputs[0]

        try:
            model = models.Model(inputs=model_inputs, outputs=model_outputs, name="SignalML_Keras_Model")
        except Exception as e:
            raise KerasGraphError(f"Keras Functional Model compilation failed: {str(e)}")

        total_params = int(model.count_params())
        trainable_params = int(sum(np.prod(p.shape) for p in model.trainable_weights))
        non_trainable_params = total_params - trainable_params

        summary = {
            "total_params": total_params,
            "trainable_params": trainable_params,
            "non_trainable_params": non_trainable_params,
            "layers": layer_summary,
            "input_shapes": [list(t.shape) for t in (model_inputs if isinstance(model_inputs, list) else [model_inputs])],
            "output_shapes": [list(t.shape) for t in (model_outputs if isinstance(model_outputs, list) else [model_outputs])]
        }

        return model, summary

    @staticmethod
    def compile_model(
        model: keras.Model,
        optimizer: str = "adam",
        learning_rate: float = 0.001,
        loss: Optional[str] = None,
        metrics: Optional[List[str]] = None
    ) -> keras.Model:
        """Compiles Keras model with specified optimizer, loss, and metrics."""
        opt_name = optimizer.lower()
        if opt_name == "sgd":
            opt = keras.optimizers.SGD(learning_rate=learning_rate)
        elif opt_name == "rmsprop":
            opt = keras.optimizers.RMSprop(learning_rate=learning_rate)
        elif opt_name == "adamw":
            opt = keras.optimizers.AdamW(learning_rate=learning_rate)
        else:
            opt = keras.optimizers.Adam(learning_rate=learning_rate)

        # Auto-detect default loss based on final layer activation if not provided
        if loss is None:
            # Check output shape
            out_shape = model.output_shape
            if isinstance(out_shape, list):
                out_shape = out_shape[0]
            if len(out_shape) == 2 and out_shape[-1] > 1:
                loss = "sparse_categorical_crossentropy"
            elif len(out_shape) == 2 and out_shape[-1] == 1:
                loss = "binary_crossentropy"
            else:
                loss = "mse"

        if metrics is None:
            metrics = ["accuracy"] if "cross" in loss else ["mae"]

        model.compile(optimizer=opt, loss=loss, metrics=metrics)
        return model
