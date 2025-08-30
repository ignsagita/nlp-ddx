## **Title:** Evaluation of Differential Diagnosis Performance with foundation model with Retrieval Augmented Generation (RAG) vs. Domain-specific Fine-tuned LLMs

## Motivation:
- An accurate differential diagnosis (DDx) is critical but challenging, involving an iterative process of interpreting clinical history, physical examination, and procedures[1].
- Large Language Models (LLMs) can assist doctors by improving diagnostic reasoning and accuracy in challenging cases, thus unlocking insights from underutilized Electronic Health Record (EHR) data.
- Baseline shows that unassisted DDx reached around ~35% top-10 accuracy in the New England Journal of Medicine (NEJM) challenging, real-world medical cases[2].
- Medical research is one of the most emerging fields, thus clinicians could upload some newest medical references/scientific journals.

## Approach:
- The solution uses LLMs to provide and broaden a clinical perspective of possible diagnosis, then the doctor will decide the next steps.
- Tech Stacks:
--> Model: BioMistral[3] and Mistral[4]
--> FAISS (vector search for RAG)
- Users can upload medical textbooks/papers in the project folder. This solution applies context-aware similarity search with token management (default: top-3 chunks).
- Use cases: evaluate diagnostic accuracy (quality management) or an educational tool for specialists.

## Results:
- Though Mistral with RAG had a more comprehensive list, it often contained more hallucinations and missed the correct diagnosis when compared to BioMistral.
- Simple RAG is not suited for developing a differential diagnosis list, as this is not a simple information retrieval task. 


## Challenges and Further Enhancement:
- Context window limitation (long cases and references).
- GPU requirement for efficient inference.
- Future: the human-in-the-loop evaluation matrix

## Quick Start:
1. Clone this repository:
   git clone https://github.com/ignsagita/nlp-ddx.git
   cd nlp-ddx
2. Install dependencies:
   pip install -r requirements.txt
3. Run the notebook:
   jupyter notebook ddx-final.ipynb

## Reference:
[1] Hirosawa T, Harada Y, Tokumasu K, Shiraishi T, Suzuki T, Shimizu T. Comparative Analysis of Diagnostic Performance: Differential Diagnosis Lists by LLaMA3 Versus LLaMA2 for Case Reports. JMIR Form Res. 2024 Nov 19;8:e64844. doi: 10.2196/64844. PMID: 39561356; PMCID: PMC11615545.
[2] McDuff, D., Schaekermann, M., Tu, T. et al. Towards accurate differential diagnosis with large language models. Nature 642, 451–457 (2025). https://doi.org/10.1038/s41586-025-08869-4
[3] Yanis Labrak, Adrien Bazoge, Emmanuel Morin, Pierre-Antoine Gourraud, Mickael Rouvier, and Richard Dufour. 2024. BioMistral: A Collection of Open-Source Pretrained Large Language Models for Medical Domains. In Findings of the Association for Computational Linguistics: ACL 2024, pages 5848–5864, Bangkok, Thailand. Association for Computational Linguistics.
[4] https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.1
