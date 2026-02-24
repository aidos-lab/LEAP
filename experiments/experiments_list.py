from dataclasses import dataclass


def to_dict(Class):
    res = {}
    for name, value in Class.__dict__.items():
        if not name.startswith("_") and not name == "to_dict":
            res[name] = value
    return res


@dataclass(frozen=True)
class ExperimentsDefinitions:
    """
    Experiments with GNN layers and different PE and ECT configurations
    """

    experiment_1 = {
        "description": "vanilla GCN",
        "model_name": "GCN",
        "conv_name": "GCNConv",
        "use_ect": False,
        "use_pos_enc": False,
    }
    experiment_2 = {
        "description": "vanilla GAT",
        "model_name": "GCN",
        "conv_name": "GATConv",
        "use_ect": False,
        "use_pos_enc": False,
    }

    ## Use ECT ##
    experiment_3 = {
        "description": "L-ECT GCN",
        "model_name": "GCN",
        "conv_name": "GCNConv",
        "use_ect": True,
        "use_pos_enc": False,
    }
    experiment_4 = {
        "description": "L-ECT GAT",
        "model_name": "GCN",
        "conv_name": "GATConv",
        "use_ect": True,
        "use_pos_enc": False,
    }
    experiment_5 = {
        "description": "L-ECT GCN",
        "model_name": "GCN",
        "conv_name": "GCNConv",
        "use_ect": True,
        "use_pos_enc": False,
        "learnable_pe": True,
    }
    experiment_6 = {
        "description": "L-ECT GAT",
        "model_name": "GCN",
        "conv_name": "GATConv",
        "use_ect": True,
        "use_pos_enc": False,
        "learnable_pe": True,
    }
    experiment_7 = {
        "description": "L-ECT GCN",
        "model_name": "GCN",
        "conv_name": "GCNConv",
        "use_ect": True,
        "use_pos_enc": False,
        "exclude_features": True,
    }
    experiment_8 = {
        "description": "L-ECT GAT",
        "model_name": "GCN",
        "conv_name": "GATConv",
        "use_ect": True,
        "use_pos_enc": False,
        "exclude_features": True,
    }

    # experiment_9 = {
    #     "description": "L-ECT GAT",
    #     "model_name": "GCN",
    #     "conv_name": "GATConv",

    #     "use_ect": True,

    #     "use_pos_enc": False,
    # }

    ## Use learnable ECT ##
    experiment_9 = {
        "description": "L-ECT GCN",
        "model_name": "GCN",
        "conv_name": "GCNConv",
        "use_ect": True,
        "learn_directions": True,
        "use_pos_enc": False,
    }
    experiment_10 = {
        "description": "L-ECT GAT",
        "model_name": "GCN",
        "conv_name": "GATConv",
        "use_ect": True,
        "learn_directions": True,
        "use_pos_enc": False,
    }

    ## Use ECT + PE ##
    experiment_11 = {
        "description": "L-ECT GCN",
        "model_name": "GCN",
        "conv_name": "GCNConv",
        "use_ect": True,
        "use_pos_enc": True,
    }
    experiment_12 = {
        "description": "L-ECT GAT",
        "model_name": "GCN",
        "conv_name": "GATConv",
        "use_ect": True,
        "use_pos_enc": True,
    }

    ## Use PE without l-ECT ##
    experiment_13 = {
        "description": "L-ECT GCN",
        "model_name": "GCN",
        "conv_name": "GCNConv",
        "use_ect": False,
        "use_pos_enc": True,
    }
    experiment_14 = {
        "description": "L-ECT GAT",
        "model_name": "GCN",
        "conv_name": "GATConv",
        "use_ect": False,
        "use_pos_enc": True,
    }
    experiment_15 = {
        "description": "L-ECT GCN",
        "model_name": "GCN",
        "conv_name": "GCNConv",
        "use_ect": False,
        "use_pos_enc": True,
        "exclude_features": True,
    }
    experiment_16 = {
        "description": "L-ECT GAT",
        "model_name": "GCN",
        "conv_name": "GATConv",
        "use_ect": False,
        "use_pos_enc": True,
        "exclude_features": True,
    }

    """
	Experiments with Transformer Encoder without GNN layers
	"""
    experiment_17 = {
        "description": "vanilla Transformer",
        "model_name": "NoMP",
        "use_ect": False,
        "use_pos_enc": False,
    }

    experiment_18 = {
        "description": "l-ECT Transformer: fixed directions",
        "model_name": "NoMP",
        "use_ect": True,
        "learn_directions": False,
        "use_pos_enc": False,
    }

    experiment_19 = {
        "description": "l-ECT Transformer: learnable directions",
        "model_name": "NoMP",
        "use_ect": True,
        "learn_directions": True,
        "use_pos_enc": False,
    }
    experiment_20 = {
        "description": "RWPE Transformer",
        "model_name": "NoMP",
        "use_ect": False,
        "learn_directions": False,
        "use_pos_enc": True,
    }

    experiment_21 = {
        "description": "l-ECT + RWPE Transformer",
        "model_name": "NoMP",
        "use_ect": True,
        "learn_directions": False,
        "use_pos_enc": True,
    }

    """
	Experiments with GNN layers and iterative L-ECT
	"""
    experiment_22 = {
        "description": "GCN Iterative L-ECT: fixed directions",
        "model_name": "GCNwithLECTallSteps",
        "conv_name": "GCNConv",
        "ect_embed_dim": 5,
        "learn_directions": False,
    }

    experiment_23 = {
        "description": "GCN Iterative L-ECT learning directions",
        "model_name": "GCNwithLECTallSteps",
        "conv_name": "GCNConv",
        "ect_embed_dim": 5,
        "learn_directions": True,
    }
    experiment_24 = {
        "description": "GAT Iterative L-ECT: fixed directions",
        "model_name": "GCNwithLECTallSteps",
        "conv_name": "GATConv",
        "ect_embed_dim": 5,
        "learn_directions": False,
    }

    experiment_25 = {
        "description": "GAT Iterative L-ECT learning directions",
        "model_name": "GCNwithLECTallSteps",
        "conv_name": "GATConv",
        "ect_embed_dim": 5,
        "learn_directions": True,
    }

    ## Experiments for HIV
    HIV_experiment_1 = {
        "description": "HIV GCN",
        "model_name": "GCN",
        "conv_name": "GCNConv",
        "hidden_dim": 128,
        "num_layers": 10,
        "batch_size": 32,
        "use_ect": False,
        "learn_directions": False,
        "use_pos_enc": False,
    }
    ## Experiments for HIV
    HIV_experiment_2 = {
        "description": "HIV GCN + l-ECT",
        "model_name": "GCN",
        "conv_name": "GCNConv",
        "hidden_dim": 128,
        "num_layers": 10,
        "batch_size": 32,
        "num_thetas": 32,
        "use_ect": True,
        "learn_directions": False,
        "use_pos_enc": False,
    }


@dataclass(frozen=True)
class Experiments4LaPE:
    ## Use PE without l-ECT ##
    experiment_13 = {
        "description": "LaPE GCN",
        "model_name": "GCN",
        "conv_name": "GCNConv",
        "use_ect": False,
        "use_pos_enc": True,
        "pe_name": "lape",
    }
    experiment_14 = {
        "description": "LaPE GAT",
        "model_name": "GCN",
        "conv_name": "GATConv",
        "use_ect": False,
        "use_pos_enc": True,
        "pe_name": "lape",
    }
    experiment_20 = {
        "description": "LaPE Transformer",
        "model_name": "NoMP",
        "use_ect": False,
        "learn_directions": False,
        "use_pos_enc": True,
        "pe_name": "lape",
    }

    experiment_aux = {
        "description": "repeat l-ect Transformer",
        "model_name": "NoMP",
        "use_ect": True,
        "learn_directions": False,
        "ect_embed_method": "conv1d",
        # "ect_embed_dim": "5"
    }


@dataclass(frozen=True)
class Experiments4alchemy_full:
    # experiment_1 = {
    # 	"description": "Vanilla GCN",
    # 	"model_name": "GCN",
    # 	"num_layers": 10,
    # 	"hidden_dim": 64,
    # 	"batch_size": 32,

    # 	"conv_name": "GCNConv",

    # 	"use_ect": False,
    # 	"learn_directions": False,

    # 	"use_pos_enc": False,
    # 	"pe_name": "rwpe",
    # }
    # experiment_2 = {
    # 	"description": "rwpe GCN",
    # 	"model_name": "GCN",
    # 	"num_layers": 10,
    # 	"hidden_dim": 64,
    # 	"batch_size": 32,

    # 	"conv_name": "GCNConv",

    # 	"use_ect": False,
    # 	"learn_directions": False,

    # 	"use_pos_enc": True,
    # 	"pe_name": "rwpe",
    # }
    # experiment_3 = {
    # 	"description": "lape GCN",
    # 	"model_name": "GCN",
    # 	"num_layers": 10,
    # 	"hidden_dim": 64,
    # 	"batch_size": 32,

    # 	"conv_name": "GCNConv",

    # 	"use_ect": False,
    # 	"learn_directions": False,

    # 	"use_pos_enc": True,
    # 	"pe_name": "lape",
    # }
    # experiment_4 = {
    # 	"description": "l-ECT GCN",
    # 	"model_name": "GCN",
    # 	"num_layers": 10,
    # 	"hidden_dim": 64,
    # 	"batch_size": 32,

    # 	"conv_name": "GCNConv",

    # 	"use_ect": True,
    # 	"learn_directions": False,

    # 	"use_pos_enc": False,
    # 	"pe_name": "rwpe",
    # }
    # experiment_5 = {
    # 	"description": "l-ECT learn directions GCN",
    # 	"model_name": "GCN",
    # 	"num_layers": 10,
    # 	"hidden_dim": 64,
    # 	"batch_size": 32,

    # 	"conv_name": "GCNConv",

    # 	"use_ect": True,
    # 	"learn_directions": True,

    # 	"use_pos_enc": False,
    # 	"pe_name": "rwpe",
    # }
    experiment_6 = {
        "description": "Vanilla GATConv",
        "model_name": "GCN",
        "num_layers": 10,
        "hidden_dim": 64,
        "batch_size": 32,
        "conv_name": "GATConv",
        "use_ect": False,
        "learn_directions": False,
        "use_pos_enc": False,
        "pe_name": "rwpe",
    }
    experiment_7 = {
        "description": "rwpe GATConv",
        "model_name": "GCN",
        "num_layers": 10,
        "hidden_dim": 64,
        "batch_size": 32,
        "conv_name": "GATConv",
        "use_ect": False,
        "learn_directions": False,
        "use_pos_enc": True,
        "pe_name": "rwpe",
    }
    experiment_8 = {
        "description": "lape GATConv",
        "model_name": "GCN",
        "num_layers": 10,
        "hidden_dim": 64,
        "batch_size": 32,
        "conv_name": "GATConv",
        "use_ect": False,
        "learn_directions": False,
        "use_pos_enc": True,
        "pe_name": "lape",
    }
    experiment_9 = {
        "description": "l-ECT GATConv",
        "model_name": "GCN",
        "num_layers": 10,
        "hidden_dim": 64,
        "batch_size": 32,
        "conv_name": "GATConv",
        "use_ect": True,
        "learn_directions": False,
        "use_pos_enc": False,
        "pe_name": "rwpe",
    }
    experiment_10 = {
        "description": "l-ECT learn directions GATConv",
        "model_name": "GCN",
        "num_layers": 10,
        "hidden_dim": 64,
        "batch_size": 32,
        "conv_name": "GATConv",
        "use_ect": True,
        "learn_directions": True,
        "use_pos_enc": False,
        "pe_name": "rwpe",
    }


@dataclass(frozen=True)
class Experiments4lECTProjection:
    ## GCN Experiments ##
    # experiment_gcn = {
    # 	"description": "GCN",
    # 	"model_name": "GCN",
    # 	"conv_name": "GCNConv",

    # 	"use_ect": True,
    # 	"learn_directions": False,

    # 	"use_pos_enc": False,
    # }

    ## GAT Experiments ##
    experiment_gat = {
        "description": "GAT",
        "model_name": "GCN",
        "conv_name": "GATConv",
        "use_ect": True,
        "learn_directions": False,
        "use_pos_enc": False,
    }
    ## NoMP Experiments ##
    # experiment_nomp = {
    # 	"description": "NoMP",
    # 	"model_name": "NoMP",
    # 	"conv_name": "GCNConv",

    # 	"use_ect": True,
    # 	"learn_directions": False,

    # 	"use_pos_enc": False,
    # }

    def to_dict():
        aux_dict = to_dict(Experiments4lECTProjection)
        methods = ["linear", "deepsets", "attn", "attn_pe", "conv1d"]
        new_dict = {}

        for key, value in aux_dict.items():
            for method in methods:
                new_key = key + "_" + method
                new_dict[new_key] = value.copy()
                new_dict[new_key]["ect_embed_method"] = method
                new_dict[new_key]["learn_directions"] = False

                new_key_ld = new_key + "_ld"
                new_dict[new_key_ld] = new_dict[new_key].copy()
                new_dict[new_key_ld]["learn_directions"] = True

        return new_dict


@dataclass(frozen=True)
class Experiments4HIV:
    experiment_1 = {
        "description": "Vanilla GAT",
        "model_name": "GCN",
        "conv_name": "GATConv",
        # "hidden_dim": 64,
        "use_ect": False,
        "learn_directions": False,
        "use_pos_enc": False,
        "pe_name": "rwpe",
    }
    experiment_2 = {
        "description": "GAT + lape",
        "model_name": "GCN",
        "conv_name": "GATConv",
        # "hidden_dim": 64,
        "use_ect": False,
        "learn_directions": False,
        "use_pos_enc": True,
        "pe_name": "lape",
    }
    experiment_3 = {
        "description": "GAT + rwpe",
        "model_name": "GCN",
        "conv_name": "GATConv",
        "hidden_dim": 64,
        "use_ect": False,
        "learn_directions": False,
        "use_pos_enc": True,
        "pe_name": "rwpe",
    }

    experiment_4 = {
        "description": "Ect NoMP",
        "model_name": "NoMP",
        "conv_name": "GCNConv",
        # "hidden_dim": 64,
        "use_ect": True,
        "learn_directions": False,
        "ect_scale": 64,
        "use_pos_enc": False,
        "pe_name": "rwpe",
    }
    experiment_5 = {
        "description": "GAT + ECT ld",
        "model_name": "GCN",
        "conv_name": "GATConv",
        # "hidden_dim": 64,
        "use_ect": True,
        "learn_directions": True,
        "ect_scale": 64,
        "use_pos_enc": False,
        "pe_name": "rwpe",
    }

    experiment_6 = {
        "description": "GAT + ECT ld low scale",
        "model_name": "GCN",
        "conv_name": "GATConv",
        "use_ect": True,
        "learn_directions": True,
        "ect_scale": 64,
        # "hidden_dim": 64,
        "use_pos_enc": False,
        "pe_name": "rwpe",
    }

    experiment_7 = {
        "description": "GCN + ECT ld low scale",
        "model_name": "GCN",
        "conv_name": "GCNConv",
        "use_ect": True,
        "learn_directions": True,
        "ect_scale": 32,
        "use_pos_enc": False,
        "pe_name": "rwpe",
    }

    experiment_8 = {
        "description": "GCN + ECT ld 16 scale",
        "model_name": "GCN",
        "conv_name": "GCNConv",
        "use_ect": True,
        "learn_directions": True,
        "ect_scale": 16,
        "use_pos_enc": False,
        "pe_name": "rwpe",
    }

    experiment_att = {
        "description": "ATT + ECT ld low scale",
        "model_name": "NoMP",
        "conv_name": "GATConv",
        "use_ect": True,
        "learn_directions": True,
        "ect_scale": 64,
        "hidden_dim": 64,
        "use_pos_enc": False,
        "pe_name": "rwpe",
    }


# EXPERIMENTS_DICT = (ExperimentsDefinitions.to_dict())
# EXPERIMENTS_LIST = [experiment for _, experiment in EXPERIMENTS_DICT.items()]
# EXPERIMENTS_LIST = [ExperimentsDefinitions.HIV_experiment_1]
# EXPERIMENTS_LIST = [Experiments4LaPE.experiment_13, Experiments4LaPE.experiment_14, Experiments4LaPE.experiment_20, Experiments4LaPE.experiment_aux]
# EXPERIMENTS_LIST = [
# 	# Experiments4alchemy_full.experiment_1,
# 	# Experiments4alchemy_full.experiment_2,
# 	# Experiments4alchemy_full.experiment_3,
# 	Experiments4alchemy_full.experiment_4,
# 	Experiments4alchemy_full.experiment_5,
# 	# Experiments4alchemy_full.experiment_6,
# 	# Experiments4alchemy_full.experiment_7,
# 	# Experiments4alchemy_full.experiment_8,
# 	# Experiments4alchemy_full.experiment_9,
# 	# Experiments4alchemy_full.experiment_10,
# 	]

# EXPERIMENTS_LIST = [exp for exp in to_dict(Experiments4alchemy_full).values()]


@dataclass(frozen=True)
class Experiments4Ablation:
    experiment_1 = {
        "description": "ablation study",
        "model_name": "NoMP",
        "use_ect": True,
        "use_pos_enc": False,
        "ect_embed_method": "conv1d",
    }

    def to_dict():
        aux_dict = to_dict(Experiments4Ablation)
        num_thetas_list = [2, 4, 8, 16, 32]
        ect_scale_list = [2, 4, 8, 16, 32, 64, 128]
        new_dict = {}

        for key, value in aux_dict.items():
            for num_thetas in num_thetas_list:
                for ect_scale in ect_scale_list:
                    new_key = key + "_" + str(num_thetas) + "_" + str(ect_scale)
                    new_dict[new_key] = value.copy()
                    new_dict[new_key]["ect_scale"] = ect_scale
                    new_dict[new_key]["num_thetas"] = num_thetas
                    new_dict[new_key]["learn_directions"] = False

                    new_key_ld = new_key + "_ld"
                    new_dict[new_key_ld] = new_dict[new_key].copy()
                    new_dict[new_key_ld]["learn_directions"] = True

        return new_dict


@dataclass(frozen=True)
class Experiments4Hops:
    experiment_1 = {
        "description": "ablation study hop number",
        "model_name": "NoMP",
        "use_ect": True,
        "use_pos_enc": False,
        "ect_embed_method": "attn_pe",
    }

    def to_dict():
        aux_dict = to_dict(Experiments4Hops)
        ect_hops_list = [(1, 2), (2,)]
        new_dict = {}

        for key, value in aux_dict.items():
            for ect_hops in ect_hops_list:
                new_key = key + "_" + str(ect_hops)
                new_dict[new_key] = value.copy()
                new_dict[new_key]["ect_hops"] = ect_hops
                new_dict[new_key]["learn_directions"] = False
                if len(ect_hops) == 2:
                    new_dict[new_key]["ect_embed_dim"] = 5

                new_key_ld = new_key + "_ld"
                new_dict[new_key_ld] = new_dict[new_key].copy()
                new_dict[new_key_ld]["learn_directions"] = True

        return new_dict


@dataclass(frozen=True)
class Experiments4PE_DIM:
    experiment_1 = {
        "description": "ablation pe dim: ect rd",
        "model_name": "NoMP",
        "use_ect": True,
        "learn_directions": False,
        "use_pos_enc": False,
        "ect_embed_method": "attn_pe",
    }
    experiment_2 = {
        "description": "ablation pe dim: ect ld",
        "model_name": "NoMP",
        "use_ect": True,
        "learn_directions": True,
        "use_pos_enc": False,
        "ect_embed_method": "attn_pe",
    }
    experiment_3 = {
        "description": "ablation pe dim: lape",
        "model_name": "NoMP",
        "use_ect": False,
        "use_pos_enc": True,
        "pe_name": "lape",
        "ect_embed_method": "attn_pe",
    }
    experiment_4 = {
        "description": "ablation pe dim: rwpe",
        "model_name": "NoMP",
        "use_ect": False,
        "use_pos_enc": True,
        "pe_name": "rwpe",
        "ect_embed_method": "attn_pe",
    }

    def to_dict():
        aux_dict = to_dict(Experiments4PE_DIM)
        dims = [2, 5, 20]
        new_dict = {}

        for key, value in aux_dict.items():
            for dim in dims:
                new_key = key + "_" + str(dim)
                new_dict[new_key] = value.copy()
                new_dict[new_key]["ect_embed_dim"] = dim
                new_dict[new_key]["pos_enc_dim"] = dim
                new_dict[new_key]["pos_enc_embed_dim"] = dim

        return new_dict


EXPERIMENTS_LIST = [val for val in Experiments4PE_DIM.to_dict().values()]
