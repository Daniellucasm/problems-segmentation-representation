# problems-segmentation-representation
# criar venv e instalar deps
pip install opencv-python scikit-image numpy pandas

# Otsu em uma imagem
python src/otsu.py --image data/industrial.png --blur 3 --out results/industrial_otsu.png

# Region Growing com 2 sementes (y,x) em JSON
python src/region_growing.py --image data/medical.png \
  --seeds '[[120,85],[200,160]]' --tau 15 --conn 8 --out results/medical_rg.png

# Aproximação poligonal
python src/poligonal.py