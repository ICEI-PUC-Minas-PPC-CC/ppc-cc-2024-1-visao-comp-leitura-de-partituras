import cv2
import numpy as np
import matplotlib.pyplot as plt
from line_detection import remove_horizontal_lines, remove_vertical_lines
import math
import pygame

# Função para inicializar o Pygame e preparar o sistema de áudio
def init_pygame():
    pygame.init()
    pygame.mixer.init()

# Função para reproduzir um som baseado no identificador
def play_sound(identifier):
    try:
        # Carrega o arquivo de som correspondente ao identificador
        pygame.mixer.music.load(f"sounds-piano/{identifier}.wav")
        pygame.mixer.music.play()
        while pygame.mixer.music.get_busy():  # Aguarda a música terminar
            pygame.time.Clock().tick(10)
    except Exception as e:
        print(f"Erro ao reproduzir o som: {str(e)}")

# Função para processar a lista de grupos e tocar os sons correspondentes
def process_and_play_sounds(groups):
    for group in groups:
        identifier = group[3]  # O identificador está na quarta posição da tupla
        print(f"Reproduzindo som para: {identifier}")
        play_sound(identifier)
        
def analyze_image(image, identifier, pentagram, groups_list):
    # Assumir que a imagem já está carregada e em escala de cinza
    if image is None:
        print("Erro: a imagem fornecida está vazia.")
        return

    # Binarizar a imagem
    _, binary_image = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)

    # Calcular a soma de pixels pretos em cada coluna
    pixel_sums = np.sum(binary_image == 0, axis=0)

    # Identificar colunas com densidade de pixels pretos alta
    max_density = np.max(pixel_sums)
    max_columns = np.where(pixel_sums == max_density)[0]

    # Encontrar todos os grupos contínuos de colunas com a densidade de pixels pretos mais alta
    diff = np.diff(max_columns)
    group_starts = np.where(diff > 1)[0] + 1
    if len(group_starts) == 0:
        # Apenas um grupo
        groups = [(max_columns[0], max_columns[-1])]
    else:
        group_starts = np.insert(group_starts, 0, 0)  # Adicionar o primeiro índice
        group_ends = np.append(group_starts[1:] - 1, len(max_columns) - 1)
        groups = [(max_columns[start], max_columns[end]) for start, end in zip(group_starts, group_ends)]

    # Verificar se os pixels pretos vão do topo até a base em Y
    full_height = binary_image.shape[0]
    valid_groups = []
    for start_col, end_col in groups:
        if np.all(np.sum(binary_image[:, start_col:end_col + 1] == 0, axis=0) == full_height):
            valid_groups.append((start_col, end_col, end_col - start_col + 1, identifier, pentagram))  # Adicionar o identificador

    # Definir um limite mínimo para a largura do grupo no eixo X
    min_width = 7  # Ajuste conforme necessário

    # Encontrar o(s) grupo(s) com a maior largura (densidade) que não são muito estreitos
    if valid_groups:
        max_width = max(valid_groups, key=lambda x: x[2])[2]
        max_groups = [grp for grp in valid_groups if grp[2] == max_width and grp[2] >= min_width]

        sorted_groups = sorted(max_groups, key=lambda x: x[0])
        # Adicionar os grupos válidos ao vetor global
        groups_list.extend(sorted_groups)
    

def group_lines_into_pentagrams(lines, image):
    # Agrupar linhas em blocos de cinco (pentagramas)
    print(f'{len(lines)} linhas.')
    pentagrams = []
    lines = sorted(lines, key=lambda x: x[1])  # Ordenar por posição vertical
    current_pentagram = []

    for line in lines:
        current_pentagram.append(line)
        if len(current_pentagram) == 5:
            pentagrams.append(current_pentagram)
            current_pentagram = []

    pentagram_images = save_and_show_pentagrams(image, pentagrams)
    return pentagram_images

def save_and_show_pentagrams(image, pentagrams):
    pentagram_images = []
    all_pentagram_groups = []
    for i, pentagram in enumerate(pentagrams):
        pentagram_groups = []
        # top_line_y = pentagram[0][1]
        # bottom_line_y = pentagram[-1][1]
        # line_spacing = pentagram[1][1] - pentagram[0][1]
        # margin = 2 * line_spacing
        # pentagram_image = image[max(top_line_y - margin, 0):bottom_line_y + margin, :]
        # pentagram_images.append(pentagram_image)
        for j in range(len(pentagram) - 1):
            top_line_y = pentagram[j][1]  # Y da linha atual
            bottom_line_y = pentagram[j + 1][1]  # Y da próxima linha para calcular o espaço
            line_spacing = bottom_line_y - top_line_y  # Espaço entre a linha atual e a próxima

            # Não há necessidade de margem aqui se quisermos apenas a linha
            pentagram_lines = image[top_line_y:bottom_line_y, :]  # Recorte da linha atual
            # plt.imshow(cv2.cvtColor(pentagram_lines, cv2.COLOR_BGR2RGB))
            # plt.title(f'Pentagrama {i}, Espaco {j}')
            # plt.show()
            
            if j == 0:
                analyze_image(pentagram_lines, 'E#', i, pentagram_groups)
            if j == 1:
                analyze_image(pentagram_lines, 'C', i, pentagram_groups)
            if j == 2:
                analyze_image(pentagram_lines, 'A', i, pentagram_groups)
            if j == 3:
                analyze_image(pentagram_lines, 'F', i, pentagram_groups)
        
        for j in range(len(pentagram) - 1):
            top_line_y = pentagram[j][1]  # Y da linha atual
            bottom_line_y = pentagram[j + 1][1]  # Y da próxima linha para calcular o espaço
            line_spacing = math.floor((bottom_line_y - top_line_y) / 2)  # Espaço entre a linha atual e a próxima
            top = top_line_y - line_spacing
            down = top_line_y + line_spacing
            # Não há necessidade de margem aqui se quisermos apenas a linha
            pentagram_image = image[top:down, :]  # Recorte da linha atual
            # plt.imshow(cv2.cvtColor(pentagram_image, cv2.COLOR_BGR2RGB))
            # plt.title(f'Pentagrama {i}, Linha {j}')
            # plt.show()
            
            if j == 0:
                analyze_image(pentagram_image, 'F#', i, pentagram_groups)
            if j == 1:
                analyze_image(pentagram_image, 'D', i, pentagram_groups)
            if j == 2:
                analyze_image(pentagram_image, 'B', i, pentagram_groups)
            if j == 3:
                analyze_image(pentagram_image, 'G', i, pentagram_groups)
            if j == 3:
                analyze_image(pentagram_image, 'E', i, pentagram_groups)
            
        
        # plt.imshow(cv2.cvtColor(pentagram_image, cv2.COLOR_BGR2RGB))
        # plt.title(f'Pentagrama {i + 1}')
        # plt.show()
        all_pentagram_groups.append(pentagram_groups)
        pentagram_groups.sort(key=lambda x: x[1])

        print(f"Grupos ordenados para o pentagrama {i+1}:")
        for group in pentagram_groups:
            print(group)
            
    print(f"Grupos ordenados:")
    for group in all_pentagram_groups:
            print(group)
            
    init_pygame()
    for group_list in all_pentagram_groups:
        process_and_play_sounds(group_list)
    pygame.quit()

    return pentagram_images  # Retorna a lista de imagens de pentagramas
