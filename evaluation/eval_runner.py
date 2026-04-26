import json
import subprocess
import re
from risk_eval import evaluate_risk

# JSON dosyasını yükle
with open("evaluation/eval_queries.json", "r", encoding="utf-8") as f:
    TEST_SET = json.load(f)

def run_query(query):
    result = subprocess.run(
        ["py", "orchestrator.py", query],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="ignore"
    )
    return result.stdout

def extract_answer(output):
    # Basit extraction (daha sonra iyileştirebiliriz)
    return output.lower()

def simple_accuracy(expected, output):
    return 1 if expected.lower() in output.lower() else 0

results = []

for item in TEST_SET:
    print("\n======================")
    print("QUERY:", item["question"])

    output = run_query(item["question"])
    answer = extract_answer(output)

    # Accuracy
    acc = simple_accuracy(item["expected_answer"], answer)

    # Risk eval (raw_data yoksa şimdilik output ile karşılaştır)
    risk = evaluate_risk(item["expected_answer"], answer)

    results.append({
        "question": item["question"],
        "expected": item["expected_answer"],
        "output": output,
        "accuracy": acc,
        "risk": risk
    })

# Kaydet
with open("evaluation/eval_results.json", "w", encoding="utf-8") as f:
    json.dump(results, f, indent=2, ensure_ascii=False)

# Özet
avg_acc = sum(r["accuracy"] for r in results) / len(results)
risk_count = sum(1 for r in results if r["risk"]["status"] == "RISK_FLAG")

print("\n===== SUMMARY =====")
print("Accuracy:", avg_acc)
print("Risk Flags:", risk_count)
