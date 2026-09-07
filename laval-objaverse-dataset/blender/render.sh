blender="${1}" 
python distribute-rendering.py --split testing --blender "$blender"
python distribute-rendering.py --split training --blender "$blender"
python distribute-rendering.py --split validation --blender "$blender"