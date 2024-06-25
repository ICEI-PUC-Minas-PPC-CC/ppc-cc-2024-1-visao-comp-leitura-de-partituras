import cv2
import matplotlib.pyplot as plt
from line_detection import detect_lines, remove_horizontal_lines
from pentagram_analysis import group_lines_into_pentagrams
from note_extraction import segment_and_save_parts


def load_and_preprocess_image(image_path):
    # Carregar a imagem
    image = cv2.imread(image_path)
    # Converter para escala de cinza
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Aplicar binarização
    _, binary = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY_INV)
    return binary, image

# def identify_notes(pentagrams):
#     # Mapear notas musicais
#     note_mapping = ['F', 'E', 'D', 'C', 'B', 'F', 'G', 'F', 'E']

#     for pentagram_lines in pentagrams:
#         print("Pentagrama:")
#         plt.imshow(image_with_lines)
#     plt.title('Linhas Horizontais Detectadas')
#     plt.show()
#         for j, line in enumerate(pentagram_lines):
#             y_position = line[1]
#             print(f"Linha {j+1} ({note_mapping[j+4]}): {y_position}")

#             if j < 4:  # Não há espaço acima da quinta linha
#                 space_y_position = (pentagram_lines[j+1][1] + y_position) / 2
#                 print(f"Espaço acima da linha {j+1} ({note_mapping[j]}): {space_y_position}")

def main(image_path):
    binary_image, original_image = load_and_preprocess_image(image_path)
    lines = detect_lines(binary_image)
    if lines:
        pentagrams = group_lines_into_pentagrams(lines, original_image)
        # identify_notes(pentagrams)
        for pentagram in pentagrams:
            gray = cv2.cvtColor(pentagram, cv2.COLOR_BGR2GRAY)
            # Aplicar binarização
            _, binary = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY_INV)
            pentagram_lines = detect_lines(binary)
            image_without_lines = remove_horizontal_lines(pentagram, pentagram_lines, line_thickness=5)
            segment_and_save_parts(image_without_lines)
    else:
        print("Nenhuma linha detectada.")

if __name__ == '__main__':
    image_path = 'partituras/partitura.jpg'
    main(image_path)
