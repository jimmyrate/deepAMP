import pandas as pd
import numpy as np

# hydrophobicity_scale = {
#     'I': 0.73, 'F': 0.61, 'V': 0.54, 'L': 0.53, 'W': 0.37,
#     'M': 0.26, 'A': 0.25, 'G': 0.16, 'C': 0.04, 'Y': 0.02,
#     'P': -0.07, 'T': -0.18, 'S': -0.26, 'H': -0.40, 'E': -0.62,
#     'N': -0.64, 'Q': -0.69, 'D': -0.72, 'K': -1.1, 'R': -1.8
# }
#
# helix_propensity_scale = {
#     # 需要从第二篇论文中获取具体的Hxi数据
#     'A': 0.00, 'R': 0.21, 'N': 0.65, 'D': 0.69, 'C': 0.68,
#     'E': 0.40, 'Q': 0.39, 'G': 1.00, 'H': 0.61, 'I': 0.41,
#     'L': 0.21, 'K': 0.26, 'M': 0.24, 'F': 0.54, 'P': 3.16,
#     'S': 0.50, 'T': 0.66, 'V': 0.61, 'W': 0.49, 'Y': 0.53
# }



# 定义从第一个文件中得到的疏水性和螺旋倾向性数据
hydrophobicity_scale = {
    'I': 0.73, 'F': 0.61, 'V': 0.54, 'L': 0.53, 'W': 0.37,
    'M': 0.26, 'A': 0.25, 'G': 0.16, 'C': 0.04, 'Y': 0.02,
    'P': -0.07, 'T': -0.18, 'S': -0.26, 'H': -0.40, 'E': -0.62,
    'N': -0.64, 'Q': -0.69, 'D': -0.72, 'K': -1.1, 'R': -1.8
}

helix_propensity_scale = {
    'A': 0.00, 'R': 0.21, 'N': 0.65, 'D': 0.69, 'C': 0.68,
    'E': 0.40, 'Q': 0.39, 'G': 1.00, 'H': 0.61, 'I': 0.41,
    'L': 0.21, 'K': 0.26, 'M': 0.24, 'F': 0.54, 'P': 3.16,
    'S': 0.50, 'T': 0.66, 'V': 0.61, 'W': 0.49, 'Y': 0.53
}

def compute_fitness(peptide_sequence):
    n = len(peptide_sequence)
    angle_offset = 100 * (np.pi / 180)  # Convert 100 degrees to radians
    cos_terms = 0
    sin_terms = 0
    exp_sum = 0

    for i, residue in enumerate(peptide_sequence):
        h = hydrophobicity_scale.get(residue, 0)
        hx = helix_propensity_scale.get(residue, 0)

        cos_terms += h * np.cos(i * angle_offset)
        sin_terms += h * np.sin(i * angle_offset)
        exp_sum += np.exp(hx)

    fitness = np.sqrt(cos_terms ** 2 + sin_terms ** 2) / exp_sum
    return fitness

# Read sequences from a text file
def read_sequences_from_txt(file_path):
    with open(file_path, 'r') as file:
        sequences = file.read().splitlines()
    return sequences

# Calculate fitness for all sequences and save to CSV
def calculate_and_save_fitness(sequences, output_csv):
    fitness_results = []
    for seq in sequences:
        fitness = compute_fitness(seq)
        fitness_results.append((seq, fitness))

    # Create a DataFrame and sort by fitness descending
    df = pd.DataFrame(fitness_results, columns=['Sequence', 'Fitness'])
    df.sort_values(by='Fitness', ascending=False, inplace=True)

    # Save to CSV
    df.to_csv(output_csv, index=False)

# Example usage
sequences = read_sequences_from_txt("/home/leo/fightinglee/AMP-Projects/Protein-Bert/data/guavanin/guavanin_first.txt")
calculate_and_save_fitness(sequences, "/home/leo/fightinglee/AMP-Projects/Protein-Bert/data/guavanin/guavanin_first.txt_with_fitness.csv")
