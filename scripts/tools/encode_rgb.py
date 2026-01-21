import sys
import os
import cv2

from ae.autoencoder import load_ae

ae = load_ae(os.environ["AE_PATH"])

def encode(path):
    image = cv2.imread(path)
    encoded = ae.encode_from_raw_image(image)
    reconstructed = ae.decode(encoded)[0]
    cv2.imwrite(path.replace(".png", "_reconstructed.png"), reconstructed)
    cv2.waitKey(1)

# Check if arguments were provided
if len(sys.argv) > 1:
    print(f"Encoding image: {sys.argv[1]}")
    encode(sys.argv[1])
else:
    print("No arguments provided.")
