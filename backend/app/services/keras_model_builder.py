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


# Custom Clustering Layer for IDEC / DEC with Student's t-distribution
@keras.saving.register_keras_serializable(package="signalml")
class ClusteringLayer(layers.Layer):
    """
    IDEC / DEC Clustering Layer:
    Transforms latent embedding space into soft cluster assignment probabilities (q_ij)
    using Student's t-distribution kernel.
    """
    def __init__(self, n_clusters: int = 10, alpha: float = 1.0, initial_weights: Any = None, **kwargs):
        super().__init__(**kwargs)
        self.n_clusters = int(n_clusters)
        self.alpha = float(alpha)
        self.initial_weights = initial_weights

    def build(self, input_shape):
        input_dim = int(input_shape[-1])
        self.clusters = self.add_weight(
            shape=(self.n_clusters, input_dim),
            initializer="glorot_uniform",
            trainable=True,
            name="clusters"
        )
        if self.initial_weights is not None:
            self.clusters.assign(self.initial_weights)
        super().build(input_shape)

    def call(self, inputs):
        # inputs shape: (batch_size, input_dim)
        # self.clusters shape: (n_clusters, input_dim)
        # q_ij = (1 + ||z_i - mu_j||^2 / alpha) ** -((alpha + 1) / 2) / sum(...)
        expanded_inputs = ops.expand_dims(inputs, axis=1)
        expanded_clusters = ops.expand_dims(self.clusters, axis=0)
        dist_sq = ops.sum(ops.square(expanded_inputs - expanded_clusters), axis=-1)

        q = 1.0 / (1.0 + (dist_sq / self.alpha))
        q = ops.power(q, (self.alpha + 1.0) / 2.0)
        q = q / ops.maximum(ops.sum(q, axis=-1, keepdims=True), 1e-10)
        q = ops.clip(q, 1e-7, 1.0 - 1e-7)
        return q

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.n_clusters)

    def get_config(self):
        config = super().get_config()
        config.update({
            "n_clusters": self.n_clusters,
            "alpha": self.alpha
        })
        return config


# Default Custom Layer Template with Turkish comments
DEFAULT_CUSTOM_LAYER_TEMPLATE = '''import keras
from keras import layers, ops

@keras.saving.register_keras_serializable(package="signalml")
class CustomSignalLayer(layers.Layer):
    """
    SignalML Özel Katman Şablonu (Keras 3 / TensorFlow Uyumlu)

    Bu sınıf, özel sinyal işleme veya derin öğrenme katmanları oluşturmak için
    standart Keras Layer arayüzünü uygular.
    """
    def __init__(self, units: int = 32, activation: str = "relu", **kwargs):
        # __init__: Katman hiperparametrelerini tanımlar ve başlatır.
        super().__init__(**kwargs)
        self.units = int(units)
        self.activation = keras.activations.get(activation)

    def build(self, input_shape):
        # build: Giriş tensörünün şekline (input_shape) göre eğitilebilir ağırlıkları (weights) oluşturur.
        input_dim = int(input_shape[-1])
        self.w = self.add_weight(
            shape=(input_dim, self.units),
            initializer="glorot_uniform",
            trainable=True,
            name="kernel"
        )
        self.b = self.add_weight(
            shape=(self.units,),
            initializer="zeros",
            trainable=True,
            name="bias"
        )
        super().build(input_shape)

    def call(self, inputs):
        # call: İleri yönlü hesaplamayı (forward pass) Keras ops ile tensörler üzerinde icra eder.
        output = ops.matmul(inputs, self.w) + self.b
        if self.activation is not None:
            output = self.activation(output)
        return output

    def compute_output_shape(self, input_shape):
        # compute_output_shape: Katmanın üreteceği çıktı tensörünün boyutunu döndürür.
        return (input_shape[0], self.units)

    def get_config(self):
        # get_config: Modelin diske kaydedilmesi (.keras / .h5) ve tekrar serileştirilebilmesi için gereklidir.
        config = super().get_config()
        config.update({
            "units": self.units,
            "activation": keras.activations.serialize(self.activation)
        })
        return config
'''


@keras.saving.register_keras_serializable(package="signalml")
class CustomSignalLayer(layers.Layer):
    """SignalML default custom layer implementation."""
    def __init__(self, units: int = 32, activation: str = "relu", **kwargs):
        super().__init__(**kwargs)
        self.units = int(units)
        self.activation = keras.activations.get(activation)

    def build(self, input_shape):
        input_dim = int(input_shape[-1])
        self.w = self.add_weight(
            shape=(input_dim, self.units),
            initializer="glorot_uniform",
            trainable=True,
            name="kernel"
        )
        self.b = self.add_weight(
            shape=(self.units,),
            initializer="zeros",
            trainable=True,
            name="bias"
        )
        super().build(input_shape)

    def call(self, inputs):
        output = ops.matmul(inputs, self.w) + self.b
        if self.activation is not None:
            output = self.activation(output)
        return output

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.units)

    def get_config(self):
        config = super().get_config()
        config.update({
            "units": self.units,
            "activation": keras.activations.serialize(self.activation)
        })
        return config


class KerasModelBuilder:
    @staticmethod
    def coerce_bool(val: Any, default: bool = False) -> bool:
        """Robust boolean coercion handling strings like 'false', '0', 'no', 'true', etc."""
        if val is None:
            return default
        if isinstance(val, bool):
            return val
        if isinstance(val, (int, float)):
            return bool(val)
        if isinstance(val, str):
            s = val.strip().lower()
            if s in ("false", "0", "no", "off", "none", ""):
                return False
            if s in ("true", "1", "yes", "on"):
                return True
        return bool(val)

    @staticmethod
    def parse_graph(graph_data: Dict[str, Any]) -> Tuple[Dict[str, Dict[str, Any]], List[Dict[str, Any]]]:
        """Parses and normalizes nodes and links from graph payload."""
        if not isinstance(graph_data, dict):
            return {}, []
        raw_nodes = graph_data.get("nodes", [])
        if isinstance(raw_nodes, dict):
            raw_nodes = list(raw_nodes.values())

        raw_links = graph_data.get("links")
        if raw_links is None:
            raw_links = graph_data.get("edges", [])
        if isinstance(raw_links, dict):
            raw_links = list(raw_links.values())

        nodes_by_id = {}
        for n in raw_nodes:
            if isinstance(n, dict):
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
                orig_id = l.get("origin_id", l.get("source", l.get("source_id", l.get("from"))))
                orig_slot = l.get("origin_slot", l.get("source_slot", l.get("from_slot", 0)))
                targ_id = l.get("target_id", l.get("target", l.get("target_id", l.get("to"))))
                targ_slot = l.get("target_slot", l.get("to_slot", 0))
                links.append({
                    "id": str(l.get("id", "")),
                    "origin_id": str(orig_id),
                    "origin_slot": int(orig_slot) if orig_slot is not None else 0,
                    "target_id": str(targ_id),
                    "target_slot": int(targ_slot) if targ_slot is not None else 0,
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
    def sanitize_layer_name(raw_name: Optional[str], fallback: str, used_names: set) -> str:
        """
        Converts a user's custom node name (props.get('custom_name') or non-generic node.get('title'))
        into a valid, unique Keras layer name (e.g. Conv1D_Ozellik_Cikarici, Birlesim_Katmani).
        """
        candidate = ""
        if raw_name:
            tr_map = str.maketrans({
                'ç': 'c', 'Ç': 'C',
                'ğ': 'g', 'Ğ': 'G',
                'ı': 'i', 'I': 'I', 'İ': 'I',
                'ö': 'o', 'Ö': 'O',
                'ş': 's', 'Ş': 'S',
                'ü': 'u', 'Ü': 'U'
            })
            s = str(raw_name).translate(tr_map)
            import re
            s = re.sub(r'[^a-zA-Z0-9_]+', '_', s)
            s = re.sub(r'_+', '_', s).strip('_')
            if s:
                candidate = s

        if not candidate:
            candidate = fallback

        unique_name = candidate
        counter = 1
        while unique_name in used_names:
            unique_name = f"{candidate}_{counter}"
            counter += 1

        used_names.add(unique_name)
        return unique_name

    @staticmethod
    def _create_keras_layer(node_type: str, props: Dict[str, Any], node_id: str, name: Optional[str] = None) -> Any:
        """Instantiates the corresponding Keras layer based on node type and properties."""
        t = KerasModelBuilder.clean_node_type(node_type)
        if not name:
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
            ret_seq = KerasModelBuilder.coerce_bool(props.get("return_sequences", False), False)
            act = props.get("activation", "tanh")
            return layers.SimpleRNN(units=units, return_sequences=ret_seq, activation=act, name=name)

        elif t == "lstm":
            units = int(props.get("units", 64))
            ret_seq = KerasModelBuilder.coerce_bool(props.get("return_sequences", False), False)
            drop = float(props.get("dropout", 0.0))
            rec_drop = float(props.get("recurrent_dropout", 0.0))
            return layers.LSTM(units=units, return_sequences=ret_seq, dropout=drop, recurrent_dropout=rec_drop, name=name)

        elif t == "gru":
            units = int(props.get("units", 64))
            ret_seq = KerasModelBuilder.coerce_bool(props.get("return_sequences", False), False)
            return layers.GRU(units=units, return_sequences=ret_seq, name=name)

        elif t == "bidirectional":
            sub_type = props.get("recurrent_type", "lstm").lower()
            units = int(props.get("units", 64))
            ret_seq = KerasModelBuilder.coerce_bool(props.get("return_sequences", False), False)
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

        # 6. IDEC Clustering & Custom Layers
        elif t in ("clustering", "clusteringlayer"):
            n_clusters = int(props.get("n_clusters", props.get("clusters", 8)))
            alpha = float(props.get("alpha", 1.0))
            return ClusteringLayer(n_clusters=n_clusters, alpha=alpha, name=name)

        elif t in ("customlayer", "custom_layer", "customsignallayer"):
            code = props.get("code")
            class_name = props.get("class_name", "CustomSignalLayer")
            if code:
                scope = {"keras": keras, "layers": layers, "ops": ops, "tf": tf, "np": np}
                try:
                    exec(code, scope)
                    layer_cls = scope.get(class_name, CustomSignalLayer)
                except Exception as ex:
                    raise KerasGraphError(f"Error compiling custom layer code: {str(ex)}")
            else:
                layer_cls = CustomSignalLayer

            kwargs = {}
            for k, v in props.items():
                if k not in ("code", "class_name", "type"):
                    kwargs[k] = v
            try:
                return layer_cls(name=name, **kwargs)
            except Exception:
                return layer_cls(name=name)

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
        used_names: set = set()

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

            # Determine custom vs fallback layer name
            custom_name = props.get("custom_name")
            title = node.get("title")
            is_generic_title = False
            if title:
                c_title = KerasModelBuilder.clean_node_type(title)
                if c_title in (clean_ntype, f"{clean_ntype}layer", f"{clean_ntype}node", ""):
                    is_generic_title = True

            raw_name = custom_name if custom_name else (title if not is_generic_title else None)
            fallback = f"{clean_ntype}_{nid}"
            layer_name = cls.sanitize_layer_name(raw_name, fallback, used_names)

            # Case A: Input node
            if clean_ntype in ("input", "inputlayer"):
                try:
                    tensor = cls._create_keras_layer(ntype, props, nid, name=layer_name)
                    input_source = props.get("input_source", "auto")
                    try:
                        tensor._input_source = input_source
                    except Exception:
                        pass
                    tensor_outputs[nid] = tensor
                    layer_summary.append({
                        "node_id": nid,
                        "layer_type": "Input",
                        "layer_name": getattr(tensor, "name", layer_name),
                        "custom_name": props.get("custom_name") or node.get("title") or getattr(tensor, "name", layer_name),
                        "trainable": False,
                        "output_shape": list(tensor.shape),
                        "params": 0,
                        "input_source": input_source
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
                layer_obj = cls._create_keras_layer(ntype, props, nid, name=layer_name)
                # Honor trainable property
                if hasattr(layer_obj, "trainable"):
                    layer_obj.trainable = cls.coerce_bool(props.get("trainable", True), True)

                # Connect layer
                if clean_ntype in ("concatenate", "add", "multiply", "average"):
                    if len(parent_tensors) < 2:
                        # If only one input connected to merge layer, pass it directly
                        out_tensor = parent_tensors[0]
                    else:
                        if clean_ntype == "concatenate":
                            ranks = [len(t.shape) for t in parent_tensors]
                            if len(set(ranks)) > 1:
                                flattened_parents = []
                                for idx, t in enumerate(parent_tensors):
                                    if len(t.shape) > 2:
                                        flat_name = f"auto_flatten_{nid}_{idx}"
                                        used_names.add(flat_name)
                                        flat_layer = layers.Flatten(name=flat_name)
                                        flat_t = flat_layer(t)
                                        flattened_parents.append(flat_t)
                                        layer_summary.append({
                                            "node_id": flat_name,
                                            "layer_type": "Flatten",
                                            "layer_name": flat_layer.name,
                                            "custom_name": flat_layer.name,
                                            "trainable": False,
                                            "output_shape": [dim if dim is not None else -1 for dim in list(flat_t.shape)],
                                            "params": 0
                                        })
                                    else:
                                        flattened_parents.append(t)
                                parent_tensors = flattened_parents
                                if hasattr(layer_obj, "axis") and (layer_obj.axis > 1 or layer_obj.axis < -1):
                                    layer_obj = layers.Concatenate(axis=-1, name=layer_obj.name)

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
                    "layer_name": layer_obj.name,
                    "custom_name": props.get("custom_name") or node.get("title") or layer_obj.name,
                    "trainable": getattr(layer_obj, "trainable", True),
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
            output_nodes = list(explicit_outputs)
            # Also include clustering nodes that are leaf nodes without outgoing connections
            for nid in topo_order:
                if len(adj[nid]) == 0 and KerasModelBuilder.clean_node_type(nodes_by_id[nid].get("type", "")) in ("clustering", "clusteringlayer"):
                    if nid not in output_nodes:
                        output_nodes.append(nid)
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
        loss: Optional[Union[str, List[str], Dict[str, str]]] = None,
        metrics: Optional[Union[List[str], Dict[str, List[str]]]] = None,
        loss_weights: Optional[Union[List[float], Dict[str, float]]] = None
    ) -> keras.Model:
        """
        Compiles Keras model with specified optimizer, loss, metrics, and loss_weights.
        Automatically handles 2-output IDEC models (e.g. loss=['kld', 'mse'] or ['mse', 'kld'] with loss_weights).
        """
        opt_name = str(optimizer).lower()
        if opt_name == "sgd":
            opt = keras.optimizers.SGD(learning_rate=learning_rate)
        elif opt_name == "rmsprop":
            opt = keras.optimizers.RMSprop(learning_rate=learning_rate)
        elif opt_name == "adamw":
            opt = keras.optimizers.AdamW(learning_rate=learning_rate)
        else:
            opt = keras.optimizers.Adam(learning_rate=learning_rate)

        # Detect multi-output models (e.g. IDEC with ClusteringLayer and Reconstruction)
        is_multi_output = isinstance(model.output, list) or (isinstance(model.output_shape, list) and len(model.output_shape) > 1)

        if is_multi_output:
            outputs = model.outputs if isinstance(model.outputs, list) else [model.output]
            output_shapes = model.output_shape if isinstance(model.output_shape, list) else [model.output_shape]

            is_idec = False
            clustering_idx = 0
            for idx, out_tensor in enumerate(outputs):
                layer = None
                if hasattr(out_tensor, "_keras_history"):
                    hist = out_tensor._keras_history
                    layer = hist[0] if isinstance(hist, tuple) else getattr(hist, "operation", None)
                l_name = getattr(layer, "name", "").lower() if layer else ""
                l_cls = layer.__class__.__name__ if layer else ""
                if l_cls == "ClusteringLayer" or "clustering" in l_name:
                    is_idec = True
                    clustering_idx = idx
                    break

            if not is_idec and len(outputs) == 2:
                # Shape heuristic: if one output is 2D and other has higher dimension or larger size
                if len(output_shapes[0]) == 2 and (len(output_shapes[1]) > 2 or output_shapes[1][-1] != output_shapes[0][-1]):
                    is_idec = True
                    clustering_idx = 0
                elif len(output_shapes[1]) == 2 and (len(output_shapes[0]) > 2 or output_shapes[0][-1] != output_shapes[1][-1]):
                    is_idec = True
                    clustering_idx = 1

            if loss is None:
                if is_idec:
                    if clustering_idx == 0:
                        loss = ["kld", "mse"]
                        if loss_weights is None:
                            loss_weights = [0.1, 1.0]
                    else:
                        loss = ["mse", "kld"]
                        if loss_weights is None:
                            loss_weights = [1.0, 0.1]
                else:
                    loss = ["mse"] * len(outputs)
            elif isinstance(loss, str):
                if "," in loss:
                    loss = [s.strip() for s in loss.split(",")]
                elif is_idec and loss in ("idec", "default"):
                    if clustering_idx == 0:
                        loss = ["kld", "mse"]
                        if loss_weights is None:
                            loss_weights = [0.1, 1.0]
                    else:
                        loss = ["mse", "kld"]
                        if loss_weights is None:
                            loss_weights = [1.0, 0.1]
                else:
                    loss = [loss] * len(outputs)

            if metrics is None:
                if is_idec:
                    metrics = [["accuracy"], ["mae"]] if clustering_idx == 0 else [["mae"], ["accuracy"]]
                else:
                    metrics = [["mae"]] * len(outputs)

            compile_kwargs = {"optimizer": opt, "loss": loss, "metrics": metrics}
            if loss_weights is not None:
                compile_kwargs["loss_weights"] = loss_weights
            model.compile(**compile_kwargs)
            return model

        # Single-output model
        if loss is None:
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
            metrics = ["accuracy"] if ("cross" in str(loss) or "kld" in str(loss)) else ["mae"]

        compile_kwargs = {"optimizer": opt, "loss": loss, "metrics": metrics}
        if loss_weights is not None:
            compile_kwargs["loss_weights"] = loss_weights
        model.compile(**compile_kwargs)
        return model

    @classmethod
    def generate_functional_python_code(
        cls,
        graph_data: Dict[str, Any],
        model_name: str = "SignalML_Model",
        task_type: str = "classification",
        dataset_metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Generates a standalone, runnable Python script implementing the neural network model
        using Keras Functional API (keras.Model(inputs=[...], outputs=[...])).
        Includes complete preprocessing pipeline (CSV loading, Input ID extraction & preservation,
        StandardScaler/MinMaxScaler, OneHotEncoder, 1D-Sequence grouping/reshape),
        Custom Layers (ClusteringLayer, PositionalEncoding, CustomSignalLayer),
        topological Functional API layer definitions with custom node names and trainable flags,
        compilation, automated training (with KMeans IDEC initialization), and CSV export with Input ID!
        """
        nodes_by_id, links = cls.parse_graph(graph_data)
        if not nodes_by_id:
            raise KerasGraphError("Graph has no nodes to export.")

        # Build in-degree and adjacency
        in_degree = {nid: 0 for nid in nodes_by_id}
        adj = defaultdict(list)
        incoming_links = defaultdict(list)

        for l in links:
            src = l["origin_id"]
            dst = l["target_id"]
            if src in nodes_by_id and dst in nodes_by_id:
                adj[src].append(dst)
                in_degree[dst] += 1
                incoming_links[dst].append(l)

        for dst in incoming_links:
            incoming_links[dst].sort(key=lambda x: x["target_slot"])

        # Topological Sort
        queue = deque([nid for nid in nodes_by_id if in_degree[nid] == 0])
        topo_order = []
        while queue:
            curr = queue.popleft()
            topo_order.append(curr)
            for nxt in adj[curr]:
                in_degree[nxt] -= 1
                if in_degree[nxt] == 0:
                    queue.append(nxt)

        if len(topo_order) < len(nodes_by_id):
            raise KerasGraphError("Cycle detected in graph! Functional API requires a Directed Acyclic Graph (DAG).")

        # Detect Custom Layers in Graph
        has_clustering = any(cls.clean_node_type(n.get("type", "")) in ("clustering", "clusteringlayer") for n in nodes_by_id.values())
        has_positional = any(cls.clean_node_type(n.get("type", "")) in ("positionalencoding", "positional_encoding") for n in nodes_by_id.values())
        custom_layer_nodes = [
            n for n in nodes_by_id.values()
            if cls.clean_node_type(n.get("type", "")) in ("customlayer", "custom_layer", "customsignallayer")
        ]

        # Gather Input and Output nodes
        input_nodes = [
            nid for nid in topo_order
            if cls.clean_node_type(nodes_by_id[nid].get("type", "")) in ("input", "inputlayer")
        ]
        if not input_nodes:
            raise KerasGraphError("Graph must have at least one Input node.")

        explicit_outputs = [
            nid for nid in topo_order
            if cls.clean_node_type(nodes_by_id[nid].get("type", "")) in ("output", "modeloutput")
        ]
        if explicit_outputs:
            output_nodes = list(explicit_outputs)
            for nid in topo_order:
                if len(adj[nid]) == 0 and cls.clean_node_type(nodes_by_id[nid].get("type", "")) in ("clustering", "clusteringlayer"):
                    if nid not in output_nodes:
                        output_nodes.append(nid)
        else:
            output_nodes = [nid for nid in topo_order if len(adj[nid]) == 0]
            if not output_nodes:
                output_nodes = [topo_order[-1]]

        # Map node id to python variable name
        var_names: Dict[str, str] = {}
        for nid in topo_order:
            node = nodes_by_id[nid]
            ntype = cls.clean_node_type(node.get("type", ""))
            safe_id = str(nid).replace("-", "_")
            var_names[nid] = f"{ntype}_{safe_id}"

        # Generate layer statements
        layer_lines: List[str] = []
        clustering_node_id = None
        clustering_n_clusters = 8

        for nid in topo_order:
            node = nodes_by_id[nid]
            ntype = node.get("type", "")
            clean_ntype = cls.clean_node_type(ntype)
            props = node.get("properties", {})
            w_vals = node.get("widgets_values")
            if isinstance(w_vals, list) and len(w_vals) > 0 and not props:
                props = {"units": w_vals[0]}
            var_name = var_names[nid]
            trainable = cls.coerce_bool(props.get("trainable", True), True)

            # Case 1: Input Layer
            if clean_ntype in ("input", "inputlayer"):
                shape_str = str(props.get("shape", props.get("input_shape", "512, 1")))
                shape_tuple = tuple(int(s.strip()) for s in shape_str.split(",") if s.strip())
                layer_lines.append(f"    # Input Node {nid}")
                layer_lines.append(f"    {var_name} = layers.Input(shape={shape_tuple}, name='{var_name}')")
                continue

            # Case 2: Intermediate / Output Layer
            in_links = incoming_links.get(nid, [])
            parent_vars = [var_names[l["origin_id"]] for l in in_links if l["origin_id"] in var_names]
            if not parent_vars:
                continue

            p_in = parent_vars[0]

            if clean_ntype == "dense":
                units = int(props.get("units", 64))
                act = props.get("activation", "relu")
                layer_lines.append(f"    {var_name} = layers.Dense(units={units}, activation='{act}', name='{var_name}', trainable={trainable})({p_in})")

            elif clean_ntype == "activation":
                act = props.get("activation", "relu")
                layer_lines.append(f"    {var_name} = layers.Activation('{act}', name='{var_name}')({p_in})")

            elif clean_ntype in ("output", "modeloutput"):
                act = props.get("activation", "linear")
                layer_lines.append(f"    {var_name} = layers.Activation('{act}', name='{var_name}')({p_in})")

            elif clean_ntype == "flatten":
                layer_lines.append(f"    {var_name} = layers.Flatten(name='{var_name}')({p_in})")

            elif clean_ntype == "reshape":
                shape_str = str(props.get("target_shape", "-1, 1"))
                target_shape = tuple(int(s.strip()) for s in shape_str.split(",") if s.strip())
                layer_lines.append(f"    {var_name} = layers.Reshape({target_shape}, name='{var_name}')({p_in})")

            elif clean_ntype == "dropout":
                rate = float(props.get("rate", 0.2))
                layer_lines.append(f"    {var_name} = layers.Dropout(rate={rate}, name='{var_name}')({p_in})")

            elif clean_ntype in ("batchnormalization", "batchnorm", "bn"):
                layer_lines.append(f"    {var_name} = layers.BatchNormalization(name='{var_name}', trainable={trainable})({p_in})")

            elif clean_ntype in ("layernormalization", "layernorm", "ln"):
                layer_lines.append(f"    {var_name} = layers.LayerNormalization(name='{var_name}', trainable={trainable})({p_in})")

            elif clean_ntype in ("unitnormalization", "unitnorm"):
                layer_lines.append(f"    {var_name} = layers.UnitNormalization(axis=-1, name='{var_name}')({p_in})")

            elif clean_ntype == "conv1d":
                filters = int(props.get("filters", 32))
                ksize = int(props.get("kernel_size", 5))
                strides = int(props.get("strides", 1))
                padding = props.get("padding", "same")
                act = props.get("activation", "relu")
                layer_lines.append(f"    {var_name} = layers.Conv1D(filters={filters}, kernel_size={ksize}, strides={strides}, padding='{padding}', activation='{act}', name='{var_name}', trainable={trainable})({p_in})")

            elif clean_ntype in ("conv1dtranspose", "deconv1d"):
                filters = int(props.get("filters", 32))
                ksize = int(props.get("kernel_size", 5))
                strides = int(props.get("strides", 2))
                padding = props.get("padding", "same")
                act = props.get("activation", "relu")
                layer_lines.append(f"    {var_name} = layers.Conv1DTranspose(filters={filters}, kernel_size={ksize}, strides={strides}, padding='{padding}', activation='{act}', name='{var_name}', trainable={trainable})({p_in})")

            elif clean_ntype == "upsampling1d":
                size = int(props.get("size", 2))
                layer_lines.append(f"    {var_name} = layers.UpSampling1D(size={size}, name='{var_name}')({p_in})")

            elif clean_ntype in ("maxpooling1d", "maxpool1d"):
                pool_size = int(props.get("pool_size", 2))
                padding = props.get("padding", "same")
                layer_lines.append(f"    {var_name} = layers.MaxPooling1D(pool_size={pool_size}, padding='{padding}', name='{var_name}')({p_in})")

            elif clean_ntype in ("averagepooling1d", "avgpool1d"):
                pool_size = int(props.get("pool_size", 2))
                padding = props.get("padding", "same")
                layer_lines.append(f"    {var_name} = layers.AveragePooling1D(pool_size={pool_size}, padding='{padding}', name='{var_name}')({p_in})")

            elif clean_ntype in ("globalmaxpooling1d", "globalmaxpool1d"):
                layer_lines.append(f"    {var_name} = layers.GlobalMaxPooling1D(name='{var_name}')({p_in})")

            elif clean_ntype in ("globalaveragepooling1d", "globalavgpool1d"):
                layer_lines.append(f"    {var_name} = layers.GlobalAveragePooling1D(name='{var_name}')({p_in})")

            elif clean_ntype == "lstm":
                units = int(props.get("units", 64))
                ret_seq = cls.coerce_bool(props.get("return_sequences", False), False)
                drop = float(props.get("dropout", 0.0))
                layer_lines.append(f"    {var_name} = layers.LSTM(units={units}, return_sequences={ret_seq}, dropout={drop}, name='{var_name}', trainable={trainable})({p_in})")

            elif clean_ntype == "gru":
                units = int(props.get("units", 64))
                ret_seq = cls.coerce_bool(props.get("return_sequences", False), False)
                layer_lines.append(f"    {var_name} = layers.GRU(units={units}, return_sequences={ret_seq}, name='{var_name}', trainable={trainable})({p_in})")

            elif clean_ntype == "simplernn":
                units = int(props.get("units", 64))
                ret_seq = cls.coerce_bool(props.get("return_sequences", False), False)
                act = props.get("activation", "tanh")
                layer_lines.append(f"    {var_name} = layers.SimpleRNN(units={units}, return_sequences={ret_seq}, activation='{act}', name='{var_name}', trainable={trainable})({p_in})")

            elif clean_ntype == "bidirectional":
                units = int(props.get("units", 64))
                ret_seq = cls.coerce_bool(props.get("return_sequences", False), False)
                sub_type = props.get("recurrent_type", "lstm").lower()
                sub_cls = "GRU" if sub_type == "gru" else "LSTM"
                layer_lines.append(f"    {var_name} = layers.Bidirectional(layers.{sub_cls}({units}, return_sequences={ret_seq}), name='{var_name}', trainable={trainable})({p_in})")

            elif clean_ntype in ("multiheadattention", "mha"):
                num_heads = int(props.get("num_heads", 4))
                key_dim = int(props.get("key_dim", 32))
                drop = float(props.get("dropout", 0.0))
                q_var = parent_vars[0]
                v_var = parent_vars[1] if len(parent_vars) > 1 else parent_vars[0]
                layer_lines.append(f"    {var_name} = layers.MultiHeadAttention(num_heads={num_heads}, key_dim={key_dim}, dropout={drop}, name='{var_name}', trainable={trainable})(query={q_var}, value={v_var})")

            elif clean_ntype == "positionalencoding":
                d_model = int(props.get("d_model", 64))
                max_len = int(props.get("max_len", 2048))
                layer_lines.append(f"    {var_name} = PositionalEncoding(d_model={d_model}, max_len={max_len}, name='{var_name}', trainable={trainable})({p_in})")

            elif clean_ntype in ("clustering", "clusteringlayer"):
                n_clusters = int(props.get("n_clusters", props.get("clusters", 8)))
                alpha = float(props.get("alpha", 1.0))
                clustering_node_id = nid
                clustering_n_clusters = n_clusters
                layer_lines.append(f"    {var_name} = ClusteringLayer(n_clusters={n_clusters}, alpha={alpha}, name='{var_name}', trainable={trainable})({p_in})")

            elif clean_ntype in ("customlayer", "custom_layer", "customsignallayer"):
                class_name = props.get("class_name", "CustomSignalLayer")
                units = int(props.get("units", 32))
                act = props.get("activation", "relu")
                layer_lines.append(f"    {var_name} = {class_name}(units={units}, activation='{act}', name='{var_name}', trainable={trainable})({p_in})")

            elif clean_ntype in ("concatenate", "add", "multiply", "average"):
                if len(parent_vars) < 2:
                    layer_lines.append(f"    {var_name} = {p_in}")
                else:
                    merge_cls = clean_ntype.capitalize()
                    axis_arg = ", axis=-1" if clean_ntype == "concatenate" else ""
                    layer_lines.append(f"    {var_name} = layers.{merge_cls}(name='{var_name}'{axis_arg})([{', '.join(parent_vars)}])")

            else:
                # Fallback activation/identity
                layer_lines.append(f"    {var_name} = layers.Identity(name='{var_name}')({p_in})")

        # Model inputs & outputs code
        inputs_list_str = ", ".join([var_names[nid] for nid in input_nodes])
        outputs_list_str = ", ".join([var_names[nid] for nid in output_nodes])

        model_code = "\n".join(layer_lines)

        # Custom layers code block
        custom_layers_code_parts: List[str] = []

        if has_clustering:
            custom_layers_code_parts.append("""
# =====================================================================
# IDEC Clustering Layer (Student's t-distribution)
# =====================================================================
@keras.saving.register_keras_serializable(package="signalml")
class ClusteringLayer(layers.Layer):
    \"\"\"
    Clustering layer converting latent representation into soft cluster assignments
    using Student's t-distribution kernel (IDEC / DEC).
    \"\"\"
    def __init__(self, n_clusters: int = 10, alpha: float = 1.0, initial_weights=None, **kwargs):
        super().__init__(**kwargs)
        self.n_clusters = int(n_clusters)
        self.alpha = float(alpha)
        self.initial_weights = initial_weights

    def build(self, input_shape):
        input_dim = int(input_shape[-1])
        self.clusters = self.add_weight(
            shape=(self.n_clusters, input_dim),
            initializer="glorot_uniform",
            trainable=True,
            name="clusters"
        )
        if self.initial_weights is not None:
            self.clusters.assign(self.initial_weights)
        super().build(input_shape)

    def call(self, inputs):
        expanded_inputs = ops.expand_dims(inputs, axis=1)
        expanded_clusters = ops.expand_dims(self.clusters, axis=0)
        dist_sq = ops.sum(ops.square(expanded_inputs - expanded_clusters), axis=-1)
        q = 1.0 / (1.0 + (dist_sq / self.alpha))
        q = ops.power(q, (self.alpha + 1.0) / 2.0)
        q = q / ops.maximum(ops.sum(q, axis=-1, keepdims=True), 1e-10)
        q = ops.clip(q, 1e-7, 1.0 - 1e-7)
        return q

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.n_clusters)

    def get_config(self):
        config = super().get_config()
        config.update({"n_clusters": self.n_clusters, "alpha": self.alpha})
        return config
""")

        if has_positional:
            custom_layers_code_parts.append("""
# =====================================================================
# Positional Encoding Layer for 1D Signals
# =====================================================================
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
""")

        if custom_layer_nodes:
            seen_classes = set()
            for cn in custom_layer_nodes:
                cprops = cn.get("properties", {})
                code = cprops.get("code")
                cname = cprops.get("class_name", "CustomSignalLayer")
                if cname in seen_classes:
                    continue
                seen_classes.add(cname)
                if code:
                    custom_layers_code_parts.append(f"\n# User Custom Layer: {cname}\n" + code.strip() + "\n")
                else:
                    custom_layers_code_parts.append(DEFAULT_CUSTOM_LAYER_TEMPLATE.strip() + "\n")

        custom_layers_block = "\n".join(custom_layers_code_parts)

        # Detect if model is IDEC (two outputs with one clustering output)
        is_idec = has_clustering and len(output_nodes) >= 2
        clustering_out_idx = 0
        if is_idec:
            for idx, out_id in enumerate(output_nodes):
                if cls.clean_node_type(nodes_by_id[out_id].get("type", "")) in ("clustering", "clusteringlayer"):
                    clustering_out_idx = idx
                    break

        dataset_filename = dataset_metadata.get("name", "dataset.csv") if dataset_metadata else "dataset.csv"

        # Code generator assembly
        script = f'''"""
SignalML Generated Python Script — Keras Functional API Model
Model Name: {model_name}
Task Type: {task_type}
Architecture: Fully Topological Functional Model with Data Preprocessing Pipeline & CSV Exporter
"""

import os
import sys
import numpy as np
import pandas as pd
import tensorflow as tf
import keras
from keras import layers, ops, models
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler, OneHotEncoder
from sklearn.cluster import KMeans

{custom_layers_block}

# =====================================================================
# 1. DATA PREPROCESSING PIPELINE
# =====================================================================
def load_and_preprocess_dataset(csv_path: str = "{dataset_filename}"):
    """
    Loads raw CSV dataset, extracts and preserves 'Input ID',
    scales numeric signal columns with StandardScaler,
    encodes categorical attributes with OneHotEncoder,
    and formats input tensors for neural network consumption.
    """
    if not os.path.exists(csv_path):
        print(f"[!] Warning: File '{{csv_path}}' not found. Generating dummy dataset for demonstration.")
        n_samples = 120
        df = pd.DataFrame(np.random.randn(n_samples, 512))
        df.insert(0, "Input ID", [f"SIG_{{i:04d}}" for i in range(n_samples)])
        df["target"] = np.random.randint(0, 4, size=n_samples)
    else:
        df = pd.read_csv(csv_path)

    print(f"[*] Raw Dataset Loaded: {{df.shape[0]}} rows, {{df.shape[1]}} columns.")

    # 1.1 Input ID Separation
    id_col = None
    for candidate in ["Input ID", "input_id", "ID", "id", "SampleID", "signal_id", "Index"]:
        if candidate in df.columns:
            id_col = candidate
            break

    if id_col is not None:
        input_ids = df[id_col].values
        features_df = df.drop(columns=[id_col])
        print(f"[*] Preserved '{{id_col}}' for identification and final CSV export.")
    else:
        input_ids = np.array([f"sample_{{i}}" for i in range(len(df))])
        features_df = df.copy()

    # 1.2 Target Label Separation (if present)
    label_col = None
    for candidate in ["label", "target", "class", "y", "cluster"]:
        if candidate in features_df.columns:
            label_col = candidate
            break

    if label_col is not None:
        y_raw = features_df[label_col].values
        features_df = features_df.drop(columns=[label_col])
        print(f"[*] Separated target label column: '{{label_col}}'")
    else:
        y_raw = None

    # 1.3 Numeric Scaling & Categorical One-Hot Encoding
    numeric_cols = features_df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = features_df.select_dtypes(exclude=[np.number]).columns.tolist()

    if numeric_cols:
        scaler = StandardScaler()
        X_num = scaler.fit_transform(features_df[numeric_cols].fillna(0.0))
    else:
        X_num = np.empty((len(features_df), 0))

    if categorical_cols:
        encoder = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
        X_cat = encoder.fit_transform(features_df[categorical_cols].fillna("missing"))
        X_all = np.hstack([X_num, X_cat])
    else:
        X_all = X_num

    # 1.4 Sequence Reshape: ensure shape matches 1D Convolutional input (samples, seq_len, 1)
    if X_all.ndim == 2:
        X_signal = np.expand_dims(X_all, axis=-1)
    else:
        X_signal = X_all

    print(f"[*] Processed Signal Tensor Shape: {{X_signal.shape}}")
    return input_ids, X_signal, y_raw


# =====================================================================
# 2. KERAS FUNCTIONAL API MODEL DEFINITION
# =====================================================================
def build_{model_name.replace(" ", "_")}() -> keras.Model:
    """
    Topologically constructs the Keras Functional API model architecture.
    """
{model_code}

    # Model Instantiation
    model = keras.Model(
        inputs=[{inputs_list_str}],
        outputs=[{outputs_list_str}],
        name="{model_name.replace(' ', '_')}"
    )
    return model


# =====================================================================
# 3. TRAINING & EVALUATION PIPELINE
# =====================================================================
def target_distribution(q: np.ndarray) -> np.ndarray:
    \"\"\"Calculates DEC / IDEC high-confidence target distribution P from soft assignments Q.\"\"\"
    weight = q ** 2 / q.sum(axis=0)
    return (weight.T / weight.sum(axis=1)).T


def main():
    print("=" * 70)
    print(" SignalML — Keras Functional API Model Pipeline")
    print("=" * 70)

    # 1. Load Data
    input_ids, X_data, y_labels = load_and_preprocess_dataset()

    # 2. Build Model
    model = build_{model_name.replace(" ", "_")}()
    model.summary()

    # 3. Compile Model
'''

        if is_idec:
            loss_weights_code = "[0.1, 1.0]" if clustering_out_idx == 0 else "[1.0, 0.1]"
            losses_code = "['kld', 'mse']" if clustering_out_idx == 0 else "['mse', 'kld']"
            script += f'''
    print("[*] Detected IDEC Model: Compiling with dual losses {losses_code} and weights {loss_weights_code}...")
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss={losses_code},
        loss_weights={loss_weights_code},
        metrics=[["accuracy"], ["mae"]]
    )

    # 4. Automated IDEC Initialization via KMeans on Bottleneck Embeddings
    print("[*] Initializing Clustering centers via KMeans on bottleneck embeddings...")
    clustering_layer = None
    for l in model.layers:
        if isinstance(l, ClusteringLayer) or l.__class__.__name__ == "ClusteringLayer":
            clustering_layer = l
            break

    if clustering_layer is not None and hasattr(clustering_layer, "input"):
        encoder = keras.Model(inputs=model.inputs, outputs=clustering_layer.input)
        bottleneck_features = encoder.predict(X_data, verbose=0)
        n_clusters = clustering_layer.n_clusters
        kmeans = KMeans(n_clusters=n_clusters, n_init=20, random_state=42)
        kmeans.fit(bottleneck_features)
        clustering_layer.set_weights([kmeans.cluster_centers_])
        print(f"[+] Successfully initialized {{n_clusters}} cluster centers!")

    # Compute Initial Target Distribution P
    initial_preds = model.predict(X_data, verbose=0)
    q_initial = initial_preds[{clustering_out_idx}] if isinstance(initial_preds, list) else initial_preds
    P_targets = target_distribution(q_initial)

    # Split Train/Val
    idx_train, idx_val = train_test_split(np.arange(len(X_data)), test_size=0.2, random_state=42)
    X_train, X_val = X_data[idx_train], X_data[idx_val]
    ids_train, ids_val = input_ids[idx_train], input_ids[idx_val]
    P_train, P_val = P_targets[idx_train], P_targets[idx_val]

    # Target sets for dual-output model
    if {clustering_out_idx} == 0:
        targets_train = [P_train, X_train]
        targets_val = [P_val, X_val]
    else:
        targets_train = [X_train, P_train]
        targets_val = [X_val, P_val]

    print("[*] Commencing IDEC Dual-Objective Training...")
    model.fit(
        X_train,
        targets_train,
        validation_data=(X_val, targets_val),
        epochs=10,
        batch_size=32,
        verbose=1
    )

    # 5. Inference & Exporting Results with Input ID
    print("[*] Running final inference on complete dataset...")
    final_preds = model.predict(X_data, verbose=0)
    q_final = final_preds[{clustering_out_idx}] if isinstance(final_preds, list) else final_preds
    assigned_clusters = np.argmax(q_final, axis=1)
    confidence = np.max(q_final, axis=1)

    export_df = pd.DataFrame({{"Input_ID": input_ids, "Assigned_Cluster": assigned_clusters, "Confidence": confidence}})
    for c in range(q_final.shape[1]):
        export_df[f"Cluster_{{c}}_Prob"] = q_final[:, c]

    output_csv_path = "{model_name.replace(' ', '_')}_idec_results.csv"
    export_df.to_csv(output_csv_path, index=False)
    print(f"[+] Results with Input ID successfully exported to '{{output_csv_path}}'!")
'''
        else:
            loss_choice = "sparse_categorical_crossentropy" if task_type == "classification" else "mse"
            metrics_choice = "['accuracy']" if task_type == "classification" else "['mae']"
            script += f'''
    print("[*] Compiling model with loss='{loss_choice}' and metrics={metrics_choice}...")
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss="{loss_choice}",
        metrics={metrics_choice}
    )

    # Split train/val
    if y_labels is not None:
        X_train, X_val, y_train, y_val, ids_train, ids_val = train_test_split(
            X_data, y_labels, input_ids, test_size=0.2, random_state=42
        )
    else:
        y_dummy = X_data
        X_train, X_val, y_train, y_val, ids_train, ids_val = train_test_split(
            X_data, y_dummy, input_ids, test_size=0.2, random_state=42
        )

    print("[*] Training model...")
    model.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        epochs=10,
        batch_size=32,
        verbose=1
    )

    # 4. Inference & CSV Export with Input ID
    print("[*] Running inference and exporting predictions...")
    preds = model.predict(X_data, verbose=0)
    export_df = pd.DataFrame({{"Input_ID": input_ids}})
    if preds.ndim > 1 and preds.shape[1] > 1:
        export_df["Predicted_Class"] = np.argmax(preds, axis=1)
        for c in range(preds.shape[1]):
            export_df[f"Class_{{c}}_Prob"] = preds[:, c]
    else:
        export_df["Prediction"] = preds.flatten()

    output_csv_path = "{model_name.replace(' ', '_')}_predictions.csv"
    export_df.to_csv(output_csv_path, index=False)
    print(f"[+] Predictions with Input ID successfully exported to '{{output_csv_path}}'!")
'''

        script += """
if __name__ == "__main__":
    main()
"""
        return script


sanitize_layer_name = KerasModelBuilder.sanitize_layer_name


