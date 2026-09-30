#Import libraries-
import os
import shutil
from pathlib import Path
from RabbitPathFinder import find_all_the_paths

#Set which Rabbit and Block we want to pull down, and where to put it->
SourceRabbitFolder = '/System/Volumes/Data/ceph/hifu/users/jbonaventura/RabbitRegistrationProj/RabbitData'
DestRabbitFolder = '/Users/jbonaventura/Downloads/RabbitHomework/InVtoBF/RabbitData'
RabbitID = "R24-104"
Block = 6

#Pulls what RabbitPathFinder.find_all_the_paths() actually uses for the
#InVivo->ExVivo->ExVivoBlock->BlockFace chain (volumes, RegTransforms, RegDataOut,
#RegDataProc incl. landmarks), plus the two folders TillingGUI.py reads directly
#for its H&E / Blockface RGB side-by-side viewer (HnE/BlockXX/Registered pngs and
#BlockFace_RGB/BlockXX/CroppedImages tiffs) - not the rest of HnE (raw/NormalizedFull).
PROGRESSION = ["InVivo", "ExVivo", "ExVivoBlock"]

def _is_junk(name):
    return name.startswith("._") or name == ".DS_Store"

def copy_file(src_path, source_root, dest_root):
    if src_path is None:
        return 0
    src_path = Path(src_path)
    if not src_path.is_file():
        return 0
    rel = src_path.relative_to(source_root)
    dest_path = Path(dest_root) / rel
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src_path, dest_path)
    copied = src_path.stat().st_size

    #Bring along LandMarker's own tiny session-state file for this volume, if it exists-
    stem = src_path.name
    for suffix in (".nii.gz", ".nii"):
        if stem.endswith(suffix):
            stem = stem[:-len(suffix)]
            break
    #Some .lm.json files are owned by another uid with owner-only permissions on ceph -
    #best-effort only, this is just LandMarker's UI session cache, not real landmark data.
    lm_src = src_path.parent / ".landmarker" / f"{stem}.lm.json"
    if lm_src.is_file():
        try:
            copied += copy_file(lm_src, source_root, dest_root)
        except PermissionError:
            print(f"  (skipping unreadable {lm_src})")
    return copied

def copy_tree(src_dir, source_root, dest_root, require_substr=None):
    src_dir = Path(src_dir)
    if not src_dir.is_dir():
        return 0
    total = 0
    for path in src_dir.rglob("*"):
        if path.is_dir() or _is_junk(path.name):
            continue
        if require_substr and require_substr not in path.name:
            continue
        rel = path.relative_to(source_root)
        dest_path = Path(dest_root) / rel
        dest_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            shutil.copy2(path, dest_path)
        except PermissionError:
            #Some .landmarker/*.lm.json session-cache files are owned by another uid,
            #owner-only permissions on ceph - best-effort only, not real registration data.
            print(f"  (skipping unreadable {path})")
            continue
        total += path.stat().st_size
    return total

def download_reg_qc_data(RabbitID, Block, SourceRabbitFolder, DestRabbitFolder):
    BlockID = "Block" + f"{Block:02d}"
    total_bytes = 0

    for moving_key in PROGRESSION:
        paths = find_all_the_paths(RabbitID, Block, SourceRabbitFolder, moving_key)

        total_bytes += copy_file(paths["Moving_FilePath"], SourceRabbitFolder, DestRabbitFolder)
        total_bytes += copy_file(paths["Fixed_FilePath"], SourceRabbitFolder, DestRabbitFolder)
        total_bytes += copy_tree(paths["RegFold"], SourceRabbitFolder, DestRabbitFolder)
        #InVivo_MR/RegDataOut is shared across every block for this rabbit - filter to ours-
        require_substr = BlockID if moving_key == "InVivo" else None
        total_bytes += copy_tree(paths["RegDataOut"], SourceRabbitFolder, DestRabbitFolder, require_substr)
        total_bytes += copy_tree(paths["RegDataProc"], SourceRabbitFolder, DestRabbitFolder)

        print(f"  {moving_key} stage done.")

    #TillingGUI.py's viewer reads these two folders directly, not via find_all_the_paths()-
    hne_registered_dir = os.path.join(SourceRabbitFolder, RabbitID, 'HnE', BlockID, 'Registered')
    bf_cropped_dir = os.path.join(SourceRabbitFolder, RabbitID, 'BlockFace_RGB', BlockID, 'CroppedImages')
    total_bytes += copy_tree(hne_registered_dir, SourceRabbitFolder, DestRabbitFolder)
    total_bytes += copy_tree(bf_cropped_dir, SourceRabbitFolder, DestRabbitFolder)
    print("  TillingGUI assets (H&E Registered + Blockface CroppedImages) done.")

    print(f"\n{RabbitID} {BlockID}: {total_bytes / 1e6:.0f} MB copied to {os.path.join(DestRabbitFolder, RabbitID)}")

if __name__ == '__main__':
    download_reg_qc_data(RabbitID, Block, SourceRabbitFolder, DestRabbitFolder)
