# clone detection logic

def detect_clones(similarity_matrix, threshold=80):
    """Detect potential clones based on similarity threshold"""
    clones = []
    n = len(similarity_matrix)

    for i in range(n):
        for j in range(i + 1, n):
            if similarity_matrix[i][j] >= threshold:
                clones.append({
                    'Text A': i + 1,
                    'Text B': j + 1,
                    'Similarity': similarity_matrix[i][j]
                })

    return clones
