import tomllib


def test_moe_venv():
    with open("pyproject.toml", "rb") as f:
        main = tomllib.load(f)
    with open("deps/moe/pyproject.toml", "rb") as f:
        moe = tomllib.load(f)

    def apply_moe_changes(pyproject_dict, moe_dict):
        for i in range(len(pyproject_dict["project"]["dependencies"])):
            if "torch" in pyproject_dict["project"]["dependencies"][i]:
                pyproject_dict["project"]["dependencies"][i] = "torch>=2.8.0"
        pyproject_dict["project"]["dependencies"].extend(["torchtitan", "blobfile>=3.0.0"])
        pyproject_dict["project"]["optional-dependencies"]["flash-attn"] = moe_dict["project"]["optional-dependencies"][
            "flash-attn"
        ]

        pyproject_dict["tool"]["uv"]["sources"]["torchtitan"] = moe_dict["tool"]["uv"]["sources"]["torchtitan"]
        pyproject_dict["tool"]["uv"]["override-dependencies"] = moe_dict["tool"]["uv"]["override-dependencies"]

    apply_moe_changes(main, moe)

    for k, v in main.items():
        assert k in moe, f"{k} not in moe"
        assert v == moe[k]
