import pandas as pd
import json

# Load Excel files
annotations = pd.read_excel("TrainingDataFINAL.xlsx")
questions = pd.read_excel("QuestionMapping.xlsx")

# Combine text segments under each label
knowledge = {}
for _, row in annotations.iterrows():
    label = row['label']
    text = str(row['text'])
    if label not in knowledge:
        knowledge[label] = []
    knowledge[label].append(text)

# Save as JSON
with open("legal_knowledge.json", "w", encoding="utf-8") as f:
    json.dump(knowledge, f, indent=2, ensure_ascii=False)

print("✅ legal_knowledge.json created successfully!")
