import cv2
import numpy as np
import matplotlib.pyplot as plt

def detect_lines(imgage):
    # Aplicar transformada de Hough para detectar linhas
    lines = cv2.HoughLinesP(imgage, 1, np.pi / 180, threshold=150, minLineLength=100, maxLineGap=10)
    if lines is None:
        print("Nenhuma linha detectada")
        return []

    # Filtrar apenas linhas horizontais
    horizontal_lines = [line[0] for line in lines if abs(line[0][1] - line[0][3]) < 10]

    # Agrupar linhas horizontais que são aproximadamente na mesma altura
    grouped_lines = []
    horizontal_lines.sort(key=lambda x: x[1])
    current_group = []

    for line in horizontal_lines:
        if not current_group:
            current_group.append(line)
        else:
            if abs(line[1] - current_group[-1][1]) < 10:
                current_group.append(line)
            else:
                grouped_lines.append(current_group)
                current_group = [line]

    if current_group:
        grouped_lines.append(current_group)

    # Calcular a média da posição y para cada grupo de linhas
    averaged_lines = []
    for group in grouped_lines:
        y_positions = [line[1] for line in group]
        avg_y = int(np.mean(y_positions))
        averaged_lines.append([group[0][0], avg_y, group[0][2], avg_y])

    # Visualizar as linhas detectadas
    image_with_lines = cv2.cvtColor(imgage, cv2.COLOR_GRAY2BGR)
    for line in averaged_lines:
        x1, y1, x2, y2 = line
        cv2.line(image_with_lines, (x1, y1), (x2, y2), (0, 0, 255), 2)
    plt.imshow(image_with_lines)
    plt.title('Linhas Horizontais Detectadas')
    plt.show()
    
    return averaged_lines

def remove_horizontal_lines(pentagram, lines, line_thickness):
    mask = np.ones(pentagram.shape[:2], dtype=np.uint8) * 255  # Cria uma máscara branca do mesmo tamanho da imagem

    # Desenhar as linhas na máscara com preto (0)
    for x1, y1, x2, y2 in lines:
        cv2.line(mask, (x1, y1), (x2, y2), 0, line_thickness)

    # Aplicar a máscara à imagem original (onde a máscara é preta, pintar de branco)
    image_result = pentagram.copy()
    image_result[mask == 0] = 255  # Pintar de branco
    
    plt.imshow(cv2.cvtColor(image_result, cv2.COLOR_BGR2RGB))
    plt.title('Imagem Final Sem Linhas Horizontais')
    plt.show()

    return image_result

def remove_vertical_lines(image):
    # Load image, grayscale, Gaussian blur, Otsu's threshold
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (3,3), 0)
    thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]

    horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (100,1))
    horizontal_mask = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, horizontal_kernel, iterations=1)
    
    # Detect vertical lines
    vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1,100))
    vertical_mask = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, vertical_kernel, iterations=1)

    # Combine masks and remove lines
    table_mask = cv2.bitwise_or(horizontal_mask, vertical_mask)
    image[np.where(table_mask==255)] = [255,255,255]

    cv2.imshow('image', image)
    cv2.waitKey()
    return
