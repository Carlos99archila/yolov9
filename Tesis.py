import detect
import predict_road
import cv2
import matplotlib.image as mpimg
import os


predict_road.predict_road("./data/images/road2.jpg")

for i in range(10):
    im = mpimg.imread("./data/images/road2.jpg")
    # cv2.imshow("Im",im)

    finalMask = mpimg.imread("./data/images/finalMask.jpg")
    # cv2.imshow("finalMask",finalMask)

    # Aplicar la máscara sobre la imagen original
    finalMaskedIm = cv2.bitwise_and(im, finalMask)
    # cv2.imwrite("./data/images/finalMaskedIm.jpg", finalMaskedIm)
    # cv2.imshow("finalMaskedIm",finalMaskedIm)

    detect.run(
        weights="./yolov9-t-converted.pt",
        source="./data/images/finalMaskedIm.jpg",
        classes=[2, 5, 7],  # Lista de enteros, no cadena
        save_txt=True,  # Asegúrate de pasar un valor booleano
        imgsz=(640, 640),
        conf_thres=0.2,
        device="cpu",
        exist_ok=True,  # Asegúrate de pasar un valor booleano
    )

    # cv2.imshow("Detect",cv2.imread('./runs/detect/exp/finalMaskedIm.jpg'))
    # cv2.waitKey(0)

    with open("./runs/detect/exp/labels/finalMaskedIm.txt", "r", encoding="utf-8") as file:
        contenido = file.readlines()
    print(len(contenido))
    os.remove("./runs/detect/exp/labels/finalMaskedIm.txt")
