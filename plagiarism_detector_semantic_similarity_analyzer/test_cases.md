# Plagiarism Detector Test Cases

## Test Case 1: Direct Copy (Should Show ~100% Similarity)
**Purpose**: Test exact plagiarism detection

**Text 1:**
```
Artificial intelligence is revolutionizing the healthcare industry by enabling faster diagnosis, personalized treatment plans, and improved patient outcomes. Machine learning algorithms can analyze vast amounts of medical data to identify patterns that human doctors might miss.
```

**Text 2:**
```
Artificial intelligence is revolutionizing the healthcare industry by enabling faster diagnosis, personalized treatment plans, and improved patient outcomes. Machine learning algorithms can analyze vast amounts of medical data to identify patterns that human doctors might miss.
```

**Text 3:**
```
Climate change poses significant challenges to global food security, requiring innovative agricultural technologies and sustainable farming practices to meet the growing population's needs.
```

**Expected Result**: Text 1 & 2 should show ~95-100% similarity (flagged as clones)

---

## Test Case 2: Heavy Paraphrasing (Should Show ~70-85% Similarity)
**Purpose**: Test semantic similarity detection with paraphrasing

**Text 1:**
```
The rapid advancement of artificial intelligence is transforming healthcare by providing quicker medical diagnoses, customized treatment approaches, and enhanced patient care. AI systems can process enormous datasets to discover medical patterns that physicians might overlook.
```

**Text 2:**
```
Machine learning technology is changing the medical field through faster disease identification, tailored therapeutic strategies, and better health outcomes. These intelligent systems analyze huge volumes of clinical information to find trends doctors could miss.
```

**Text 3:**
```
Social media platforms have fundamentally altered how people communicate, share information, and maintain relationships in the digital age.
```

**Expected Result**: Text 1 & 2 should show ~70-85% similarity (likely flagged as clones)

---

## Test Case 3: Partial Overlap (Should Show ~40-60% Similarity)
**Purpose**: Test detection of partial plagiarism

**Text 1:**
```
The benefits of renewable energy sources include reduced carbon emissions, energy independence, and long-term cost savings. Solar and wind power are becoming increasingly affordable and efficient.
```

**Text 2:**
```
Solar and wind power are becoming increasingly affordable and efficient. However, the intermittent nature of these energy sources requires advanced battery storage solutions and smart grid technology to ensure reliable power supply.
```

**Text 3:**
```
Electric vehicles are gaining popularity due to environmental concerns and government incentives. Tesla has been a pioneer in this market, pushing traditional automakers to invest in electric technology.
```

**Expected Result**: Text 1 & 2 should show ~40-60% similarity (moderate overlap)

---

## Test Case 4: Different Topics (Should Show <30% Similarity)
**Purpose**: Test that unrelated texts have low similarity

**Text 1:**
```
Quantum computing represents a paradigm shift in computational power, utilizing quantum mechanical phenomena like superposition and entanglement to process information exponentially faster than classical computers.
```

**Text 2:**
```
The Mediterranean diet, rich in olive oil, fish, vegetables, and whole grains, has been associated with numerous health benefits including reduced risk of heart disease and improved cognitive function.
```

**Text 3:**
```
Professional basketball requires exceptional athleticism, strategic thinking, and teamwork. Players must master fundamental skills like shooting, dribbling, and defensive positioning while adapting to fast-paced game situations.
```

**Expected Result**: All pairs should show <30% similarity (no clones detected)

---

## Test Case 5: Synonym Substitution (Should Show ~60-80% Similarity)
**Purpose**: Test detection of synonym-based plagiarism

**Text 1:**
```
Students often struggle with time management during their academic journey. Effective study habits, proper scheduling, and stress management techniques are essential for academic success and personal well-being.
```

**Text 2:**
```
Pupils frequently have difficulty with time organization throughout their educational experience. Efficient learning practices, appropriate planning, and anxiety control methods are crucial for scholastic achievement and individual wellness.
```

**Text 3:**
```
The stock market experienced significant volatility last quarter due to inflation concerns and geopolitical tensions. Investors are advised to diversify their portfolios and maintain a long-term perspective.
```

**Expected Result**: Text 1 & 2 should show ~60-80% similarity (likely flagged as clones)

---

## How to Test:

1. **Run each test case separately** in your plagiarism detector
2. **Set threshold to 80%** for clone detection
3. **Try different embedding models** (MiniLM vs MPNet) to compare results
4. **Document the results** for each model

## Expected Behavior Summary:

| Test Case | Expected Similarity | Should Be Flagged? | Purpose |
|-----------|-------------------|------------------|---------|
| Test 1 | ~95-100% | ✅ Yes | Direct copying |
| Test 2 | ~70-85% | ✅ Yes | Paraphrasing |
| Test 3 | ~40-60% | ❌ No | Partial overlap |
| Test 4 | <30% | ❌ No | Different topics |
| Test 5 | ~60-80% | ✅ Yes | Synonym substitution |

## Additional Testing Tips:

1. **Test with different thresholds** (60%, 70%, 80%, 90%) to see how it affects detection
2. **Compare model performance** - MPNet should generally be more accurate than MiniLM
3. **Try mixing languages** if you want to test cross-language detection
4. **Test with very short texts** vs **very long texts** to see performance differences
5. **Add more texts** (4-5 per test) to see the full similarity matrix

These test cases will help you understand how well your semantic similarity detector works across different types of plagiarism scenarios!