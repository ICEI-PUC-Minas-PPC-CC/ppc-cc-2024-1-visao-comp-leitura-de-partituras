import numpy as np
import cv2
import matplotlib.pyplot as plt
from predict_note import classify_image

def converter_para_preto_e_branco(imagem):
    # Converter a imagem para escala de cinza, se ainda não estiver
    if len(imagem.shape) == 3:
        imagem = cv2.cvtColor(imagem, cv2.COLOR_BGR2GRAY)

    # Aplicar limiarização para converter para preto e branco
    _, imagem_bin = cv2.threshold(imagem, 127, 255, cv2.THRESH_BINARY)

    return imagem_bin

def fill_spaces(imagem):
    # Verificar se a imagem está em escala de cinza ou colorida
    if len(imagem.shape) == 3:
        gray = cv2.cvtColor(imagem, cv2.COLOR_BGR2GRAY)
    else:
        gray = imagem

    # Mostrar imagem em escala de cinza
    cv2.imshow('Imagem em escala de cinza', gray)
    cv2.waitKey(0)

    # Aplicar detecção de bordas Canny
    canny = cv2.Canny(gray, 50, 150, apertureSize=3)
    # Mostrar resultado do Canny
    cv2.imshow('Detecção de bordas Canny', canny)
    cv2.waitKey(0)

    # Dilatar as bordas para preencher os espaços
    kernel = np.ones((7,7), np.uint8)
    dilated = cv2.dilate(canny, kernel, iterations=1)
    # Mostrar imagem dilatada
    cv2.imshow('Bordas dilatadas', dilated)
    cv2.waitKey(0)

    # Encontrar contornos
    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Preencher os contornos na imagem original com branco
    cv2.drawContours(imagem, contours, -1, (255, 255, 255), thickness=cv2.FILLED)
    # Mostrar imagem com contornos preenchidos
    cv2.imshow('Imagem com contornos preenchidos', imagem)
    cv2.waitKey(0)

    # Converter para preto e branco
    imagem_bin = converter_para_preto_e_branco(imagem)

    # Inverter as cores para ter fundo branco e símbolos pretos
    imagem_invertida = cv2.bitwise_not(imagem_bin)
    cv2.imshow('Imagem Invertida', imagem_invertida)
    cv2.waitKey(0)

    return imagem_invertida

def group_close_boxes(boxes, min_distance):
    # Agrupar caixas com base na distância mínima verticalmente
    grouped = []
    for box in sorted(boxes, key=lambda b: b[0]):  # Ordenar por posição x
        if not grouped:
            grouped.append([box])
        else:
            if box[0] - (grouped[-1][-1][0] + grouped[-1][-1][2]) < min_distance:
                grouped[-1].append(box)
            else:
                grouped.append([box])
    return grouped

def segment_and_save_parts(image, min_distance=6):
    # Garantir que a imagem está em escala de cinza
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Inverter a imagem para detectar os contornos das figuras brancas em fundo preto
    _, image = cv2.threshold(image, 128, 255, cv2.THRESH_BINARY_INV)

    # Encontrar contornos
    contours, _ = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Lista para armazenar os bounding boxes dos contornos
    bounding_boxes = [cv2.boundingRect(contour) for contour in contours]

    # Agrupar contornos que estão próximos um do outro verticalmente
    grouped_boxes = group_close_boxes(bounding_boxes, min_distance)

    # Extrair cada grupo como uma imagem separada
    parts = []
    for group in grouped_boxes:
        # Calcular o retângulo delimitador mínimo para o grupo
        x_min = min([box[0] for box in group])
        y_min = min([box[1] for box in group])
        x_max = max([box[0] + box[2] for box in group])
        y_max = max([box[1] + box[3] for box in group])

        # Cortar a imagem
        part = image[y_min:y_max, x_min:x_max]
        nova = fill_spaces(part)
        predicted_note = classify_image(nova)
        
        # Visualizar a parte e a classificação
        plt.imshow(part, cmap='gray')
        plt.title(f'Predicted Note: {predicted_note}')
        plt.show()
        