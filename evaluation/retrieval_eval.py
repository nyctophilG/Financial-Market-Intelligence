def precision_at_k(retrieved_docs, relevant_doc, k=3):
    top_k = retrieved_docs[:k]
    correct = sum(1 for doc in top_k if doc == relevant_doc)
    return correct / k

def hit_rate_at_k(retrieved_docs, relevant_doc, k=3):
    return 1 if relevant_doc in retrieved_docs[:k] else 0

if __name__ == "__main__":
    retrieved_docs = ["AAPL_Q3_2023", "TSLA_Q4_2023", "AAPL_Q3_2023"]
    relevant_doc = "AAPL_Q3_2023"

    print("Precision@3:", precision_at_k(retrieved_docs, relevant_doc, k=3))
    print("Hit Rate@3:", hit_rate_at_k(retrieved_docs, relevant_doc, k=3))
