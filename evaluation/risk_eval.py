import re

def extract_numbers(text):
    return re.findall(r"\d+(?:\.\d+)?(?:,\d{3})*", text)

def evaluate_risk(raw_data, analyst_output):
    raw_numbers = extract_numbers(raw_data)
    output_numbers = extract_numbers(analyst_output)

    mismatches = [n for n in output_numbers if n not in raw_numbers]
    risk_flags = []

    if mismatches:
        risk_flags.append("numerical_error")

    return {
        "status": "RISK_FLAG" if risk_flags else "APPROVED",
        "risk_flags": risk_flags,
        "mismatched_numbers": mismatches,
        "confidence": max(0.0, 1.0 - 0.3 * len(risk_flags))
    }

if __name__ == "__main__":
    raw_data = "Apple Q3 2023: Total net sales were $81.8 billion."
    analyst_output = "Apple generated $123.0 billion in total net sales."

    print(evaluate_risk(raw_data, analyst_output))
