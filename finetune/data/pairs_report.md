# Fine-tuning pairs - filter report

Chunks with questions: 448 (LLM errors: 0). Candidate questions: 896.
**Kept: 683/896 (76%)** -> train 615, validation 68.

| Dropped because | count |
|---|---|
| copies a phrase from its passage | 181 |
| refers to 'the passage' | 21 |
| near-duplicate | 5 |
| too similar to a gold question | 4 |
| passage not in BM25 or dense top 50 | 2 |

## Sample of kept questions (20, spread over the list)

Check: would a real engineer or lawyer ask this? Is it answered by the passage? Is the negative really wrong?

- **How can the submission period for the Performance Guarantee be extended, and what are the requirements for doing so?**  
  passage: CPWD GCC 2020, Clause 1, p. 12 (PDF 14)  
  negative: CPWD GCC 2020, Bank Guarantee Bond Forms, pp. 99-100 (PDF 101-102)
- **What does the notation Ae - Ar represent in the context of wind load calculations?**  
  passage: IS 875 (Part 3):1987, Cl. 2-2.1, pp. 5-6 (PDF 9-10)  
  negative: IS 875 (Part 3):1987, Cl. 6.3, pp. 36-37 (PDF 40-41)
- **What actions must the contractor take if the work is delayed due to force majeure, abnormally bad weather, or strikes?**  
  passage: CPWD GCC 2020, Clause 5, pp. 18-19 (PDF 20-21)  
  negative: CPWD GCC 2020, Integrity Agreement, item 1, p. 59 (PDF 61)
- **How is the increase or decrease in labor costs calculated, and what factors are considered in this calculation?**  
  passage: CPWD GCC 2020, Clause 10C, p. 26 (PDF 28)  
  negative: CPWD GCC 2020, Clause 10CC, pp. 30-31 (PDF 32-33)
- **How is the refund of the security deposit structured for road work projects?**  
  passage: CPWD GCC 2020, Clause 17, p. 38 (PDF 40)  
  negative: CPWD GCC 2020, Clause 1A, p. 13 (PDF 15)
- **What are the qualifications and certification requirements for the skilled and semi-skilled tradesmen the contractor must deploy?**  
  passage: CPWD GCC 2020, Clause 19K, p. 45 (PDF 47)  
  negative: CPWD GCC 2020, Clause 32, p. 52 (PDF 54)
- **What are the conditions under which a contractor can construct temporary wells on government land for construction purposes?**  
  passage: CPWD GCC 2020, Clause 30A, p. 51 (PDF 53)  
  negative: CPWD GCC 2020, Conditions of Contract, items 1-2, p. 9 (PDF 11)
- **Can a contractor's tender be considered invalid if it does not specify a percentage above or below the total estimated cost in both figures and words?**  
  passage: CPWD GCC 2020, General Rules & Directions, items 4-4A, p. 6 (PDF 8)  
  negative: CPWD GCC 2020, General Rules & Directions, items 7-8, p. 7 (PDF 9)
- **What safety measures must be taken before workers enter sewers and manholes?**  
  passage: CPWD GCC 2020, CPWD Safety Code, item 8 (cont.), pp. 64-65 (PDF 66-67)  
  negative: CPWD GCC 2020, CPWD Safety Code, items 8-9, pp. 65-66 (PDF 67-68)
- **What are the requirements for the contractor to maintain the attendance of each workman and where should the attendance card be kept?**  
  passage: CPWD GCC 2020, Contractor's Labour Regulations, items 6-13, pp. 76-77 (PDF 78-79)  
  negative: CPWD GCC 2020, Contractor's Labour Regulations, item 6, pp. 75-76 (PDF 77-78)
- **How can individuals with a Diploma and at least ten years of relevant experience in a reputed construction company be considered equivalent to Graduate Engineers for deployment purposes?**  
  passage: CPWD GCC 2020, Proforma of Schedules, item 32 (cont.), p. 106 (PDF 108)  
  negative: IS 875 (Part 3):1987, Table 1 (Note), p. 11 (PDF 15)
- **Can disregarding orders from superior courts in India be considered a violation of fundamental policy?**  
  passage: Ssangyong v NHAI (SC 2019), ¶ 13 (part 2/2), p. 17  
  negative: Ssangyong v NHAI (SC 2019), ¶ 20 (part 1/6), p. 31
- **How did the 1996 Act get amended in 2015?**  
  passage: Ssangyong v NHAI (SC 2019), ¶ 21 (part 1/3), pp. 35-36  
  negative: CPWD GCC 2020, Clause 25, p. 48 (PDF 50)
- **What notable court decisions have rejected defenses of arbitral tribunals acting ultra petita?**  
  passage: Ssangyong v NHAI (SC 2019), ¶ 39 (part 7/8), p. 64  
  negative: Ssangyong v NHAI (SC 2019), ¶ 40 (part 8/8), pp. 68-69
- **What does this imply about the applicability of Section 34(2)(a)(iv) in this case?**  
  passage: Ssangyong v NHAI (SC 2019), ¶ 47, pp. 86-87  
  negative: Ssangyong v NHAI (SC 2019), ¶ 10 (part 1/6), p. 11
- **How does the classification of terrain affect the assessment of wind speeds and the planning of structures, particularly in areas with varying degrees of obstruction?**  
  passage: IS 875 (Part 3):1987, Cl. 5.3.2.1 (part 1/2), p. 8 (PDF 12)  
  negative: IS 875 (Part 3):1987, Cl. 5.3-5.3.2, p. 8 (PDF 12)
- **How should the forces be considered for a duopitch canopy to ensure it can withstand them effectively?**  
  passage: IS 875 (Part 3):1987, Table 8 (part 3/3), pp. 20-21 (PDF 24-25)  
  negative: IS 875 (Part 3):1987, Cl. 6.2.2.3-6.2.2.4 (part 1/2), p. 15 (PDF 19)
- **What are the force coefficients for buildings with a height-to-width ratio of 1.2 and a shape factor of 1.4?**  
  passage: IS 875 (Part 3):1987, Table 23 (part 3/4), pp. 41-42 (PDF 45-46)  
  negative: IS 875 (Part 3):1987, Table 28, p. 46 (PDF 50)
- **How do you determine the background factor B for a structure, and what factors influence its value?**  
  passage: IS 875 (Part 3):1987, Cl. 8.3 (part 1/2), pp. 49-50 (PDF 53-54)  
  negative: IS 875 (Part 3):1987, Cl. 5.3-5.3.2, p. 8 (PDF 12)
- **Could you provide the new reference that replaces 'See also Appendix C' in Table 23 of IS 875 (Part 3)?**  
  passage: IS 875 (Part 3):1987, Amendment No. 1, item 2, PDF p. 64  
  negative: IS 875 (Part 3):1987, Cl. 6.3.3.2 (part 1/2), p. 39 (PDF 43)

## Examples of dropped questions

- *copies a phrase from its passage*: What specific details must a contractor provide to the Engineer-in-Charge by the 4th and 19th of every month regarding the work they are doing? [cpwd:clause-19d:1]
- *copies a phrase from its passage*: How does the authority mentioned in Schedule F determine the fine for a contractor who fails to submit accurate statements as per Clause 19D? [cpwd:clause-19d:1]
- *copies a phrase from its passage*: How often should the inside walls of the kitchen be lime-washed according to the model rules? [cpwd:model-rules-health-sanitation:p73-7]
- *refers to 'the passage'*: How does the passage address the compensation for losses due to hostilities or warlike operations for materials and tools not on the site? [cpwd:clause-39:2]
- *refers to 'the passage'*: How does the passage suggest maintaining the integrity of the public policy ground in the context of section 28(1)? [ssangyong:p-18-part-4-5]
- *refers to 'the passage'*: What broader recommendations does the Commission make to ensure the legitimacy of court decisions, as discussed in the passage? [ssangyong:p-18-part-4-5]
- *near-duplicate*: How is the lowest tender decided when all contractors refuse to submit revised offers? [cpwd:general-rules-directions:p8-4]
- *near-duplicate*: What is the purpose of the Bank Guarantee in this agreement? [cpwd:bank-guarantee-bond-forms:p103-6]
- *near-duplicate*: How does the Bank's liability under this guarantee differ from the contractor's obligations? [cpwd:bank-guarantee-bond-forms:p103-6]
- *too similar to a gold question*: What are the conditions for submitting a Performance Guarantee in this contract? [cpwd:clause-1:1] (0.85 to gold: What is the amount of the Performance Guarantee the contractor must submit under Clause 1?)
- *too similar to a gold question*: What are the force coefficients for single frames with either all flat-sided members or all circular members, and how do they vary based on the member type, diameter, design wind speed, and solidity ratio? [is875:cl-6-3-3-3-6-3-3-4] (0.92 to gold: What is the force coefficient for a single frame with solidity ratio 0.2 and flat-sided members?)
- *too similar to a gold question*: Can a court overturn an arbitral award merely because it believes justice was not served? [ssangyong:p-48-part-2-2] (0.87 to gold: Can a court overturn an arbitration award just because it feels justice wasn't done?)
- *passage not in BM25 or dense top 50*: How did the court in this case, Ssangyong v NHAI, likely approach the legal arguments presented by both parties, particularly focusing on the nature of the dispute and the applicable laws? [ssangyong:case-details]
- *passage not in BM25 or dense top 50*: Can you provide the wind angle distribution for buildings with a height less than 2 meters, as specified in IS 875 (Part 3)? [is875:table-6-part-1-2]
