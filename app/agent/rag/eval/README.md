# RAG Evaluation Starter

## 1. Files

- `evals/datasets/rag_eval.json`: put your golden test set here.
- `evals/rag_adapter.py`: connect the evaluator to the project's existing retriever and generator.
- `evals/run_eval.py`: computes retrieval metrics and optionally calls Ragas.
- `evals/results/`: output files are written here (create automatically).

## 2. Dataset schema

The evaluator accepts the test schema you already use:

```json
{
  "cases": [
    {
      "id": "q_001",
      "type": "direct",
      "query": "Example question",
      "expected_chunk_ids": ["会员权益#01-00"],
      "reference": "Reference answer",
      "grading_notes": ["Required fact A", "Required fact B"],
      "answerable": true,
      "difficulty": "easy"
    }
  ]
}
```

`reference` is required for useful Ragas answer-quality evaluation. `grading_notes` are retained in traces for human/LLM-judge checks but are not automatically scored by this script.

## 3. Connect to the project

Implement `retrieve()` and `generate()` in `evals/rag_adapter.py`.

The retriever should return the *actual ranked chunks* from the app's existing retrieval code. Each chunk must have a stable ID. Update `_chunk_id()` in `run_eval.py` if the project uses another metadata key.

Do not make an independent second implementation of retrieval for evaluation: that can make the baseline differ from production.

If the existing app exposes only an end-to-end `ask(query)` method, first instrument it to capture the retrieved chunks and response in one run. Avoid calling retrieval once for metrics and then asking the full pipeline to retrieve again, because results may differ.

## 4. Run retrieval metrics only

After the adapter is implemented:

```bash
python -m evals.run_eval \
  --dataset evals/datasets/rag_eval.json \
  --output-dir evals/results/baseline \
  --top-k 5
```

Metrics:
- `hit@K`: at least one expected chunk appears in top K.
- `recall@K`: fraction of expected chunks found in top K.
- `precision@K`: number of expected chunks found divided by K.
- `mrr@K`: reciprocal rank of the first expected chunk.

Cases with an empty `expected_chunk_ids` list are excluded from these retrieval averages. Evaluate abstention separately for those cases.

## 5. Run Ragas

Install dependencies compatible with the script's legacy `evaluate()` API:

```bash
pip install -r requirements.txt
```

Then:

```bash
export OPENAI_API_KEY="..."
python -m evals.run_eval \
  --dataset evals/datasets/rag_eval.json \
  --output-dir evals/results/baseline \
  --top-k 5 \
  --with-ragas
```

Ragas metrics in this template:
- Context Precision
- Context Recall
- Faithfulness
- Answer Correctness

The Ragas run uses answerable cases with a non-empty `reference`. It does not evaluate unanswerable/abstention cases. Also check the metric names and imports if you use a different Ragas version.

## 6. Output files

- `traces.json`: query, expected/retrieved IDs, contexts, response, metrics, latency.
- `summary.json`: aggregate retrieval metrics and mean latency.
- `ragas_per_case.csv`: per-case Ragas scores when supported by the installed version.

## 7. Important evaluation caveats

1. Chunk IDs in `expected_chunk_ids` must exactly match the IDs returned by the current chunker/index.
2. For multi-chunk questions, list all chunks required for a complete answer; otherwise Recall@K may be misleading.
3. Do not treat an empty gold list as a perfect retrieval result.
4. Ragas LLM-judge scores can vary; keep model/configuration fixed and manually inspect important failures.
5. Record the knowledge-base version, embedding model, chunking config, top K, and prompt/model version alongside each experiment.
