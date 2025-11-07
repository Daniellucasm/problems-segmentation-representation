# problems-segmentation-representation
# criar venv e instalar deps
pip install opencv-python scikit-image numpy pandas

# Otsu em uma imagem
python src/otsu.py --image data/industrial.jpg --blur 3 --out results/industrial_otsu.png

# Region Growing em uma imagem
python src/region_growing.py --image data/medica.jpg \
  --seeds '[[120,85],[200,160]]' --tau 15 --conn 8 --out results/medical_rg.png

# Aproximação poligonal
python src/poligonal.py
