from pathlib import Path
import mkdocs_gen_files

root = Path(__file__).parent / "python" / "traj_dist_rs"

with mkdocs_gen_files.open("api_reference.md", "w") as fd:
    fd.write("::: traj_dist_rs")

nav = mkdocs_gen_files.Nav()
nav["API Reference"] = "api_reference.md"
