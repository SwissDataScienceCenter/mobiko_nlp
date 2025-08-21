import json
from collections import Counter, defaultdict
from typing import Dict, List, Tuple, Set
import argparse


def calculate_overlap(span1_start: int, span1_end: int, span2_start: int, span2_end: int) -> float:
    """Calculate the overlap ratio between two spans."""
    overlap_start = max(span1_start, span2_start)
    overlap_end = min(span1_end, span2_end)

    if overlap_start >= overlap_end:
        return 0.0

    overlap_length = overlap_end - overlap_start
    span1_length = span1_end - span1_start
    span2_length = span2_end - span2_start

    # Return the overlap ratio relative to the smaller span
    min_length = min(span1_length, span2_length)
    return overlap_length / min_length if min_length > 0 else 0.0


def spans_match(span1: Dict, span2: Dict, overlap_threshold: float = 0.5) -> bool:
    """Check if two spans match based on overlap threshold."""
    overlap = calculate_overlap(
        span1['start_char'], span1['end_char'],
        span2['start_char'], span2['end_char']
    )
    return overlap >= overlap_threshold


def load_spans(path):
    spans_by_key = defaultdict(list)
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            r = json.loads(line)
            for i, sentence in enumerate(r['sentences']):
                key = (r["doc_id"], i)
                spans_by_key[key].append(sentence)
    return spans_by_key


def calculate_overlap_metrics(gold_spans, predicted_spans, sentences, overlap_threshold=0.5):
    """
    Calculate overlap metrics comparing spans per sentence and accumulating results.
    Uses overlap-based matching instead of exact position matching.

    Args:
        gold_spans: Dict[sentence_id, List[Dict]] or List[Dict] - gold standard spans
        predicted_spans: Dict[sentence_id, List[Dict]] or List[Dict] - predicted spans
        overlap_threshold: Minimum overlap ratio for spans to be considered matching

    Returns:
        Dict containing accumulated precision, recall, F1, and per-sentence metrics
    """
    # Handle case where spans are not grouped by sentence
    if isinstance(gold_spans, list):
        gold_spans = {0: gold_spans}
    if isinstance(predicted_spans, list):
        predicted_spans = {0: predicted_spans}

    total_tp = 0
    total_fp = 0
    total_fn = 0
    per_sentence_metrics = {}

    # Get all sentence IDs from both gold and predicted
    all_sentence_ids = set(gold_spans.keys()) | set(predicted_spans.keys())

    for sentence_id in all_sentence_ids:
        gold_sent_spans = gold_spans.get(sentence_id, [])
        pred_sent_spans = predicted_spans.get(sentence_id, [])

        # Find matching spans using overlap threshold
        matched_gold = set()
        matched_pred = set()

        for i, pred_span in enumerate(pred_sent_spans):
            for j, gold_span in enumerate(gold_sent_spans):
                if j not in matched_gold and spans_match(pred_span, gold_span, overlap_threshold):
                    matched_gold.add(j)
                    matched_pred.add(i)
                    break  # Each predicted span can only match one gold span

        # Calculate metrics for this sentence
        tp = len(matched_pred)
        fp = len(pred_sent_spans) - tp
        fn = len(gold_sent_spans) - len(matched_gold)

        # Accumulate totals
        total_tp += tp
        total_fp += fp
        total_fn += fn

        # Calculate per-sentence metrics
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

        per_sentence_metrics[sentence_id] = {
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'tp': tp,
            'fp': fp,
            'fn': fn
        }

    # Calculate overall accumulated metrics
    overall_precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
    overall_recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
    overall_f1 = 2 * overall_precision * overall_recall / (overall_precision + overall_recall) if (overall_precision + overall_recall) > 0 else 0.0

    return {
        'overall': {
            'precision': overall_precision,
            'recall': overall_recall,
            'f1': overall_f1,
            'total_tp': total_tp,
            'total_fp': total_fp,
            'total_fn': total_fn
        },
        'per_sentence': per_sentence_metrics
    }


def print_evaluation_results(results: Dict):
    """Print evaluation results in a readable format."""
    print("=" * 60)
    print("EVALUATION RESULTS")
    print("=" * 60)

    # Overall metrics
    overall = results['overall']
    print(f"\nOVERALL METRICS:")
    print(f"Precision: {overall['precision']:.3f}")
    print(f"Recall:    {overall['recall']:.3f}")
    print(f"F1-Score:  {overall['f1']:.3f}")
    print(f"True Positives:  {overall['total_tp']}")
    print(f"False Positives: {overall['total_fp']}")
    print(f"False Negatives: {overall['total_fn']}")



def main():
    parser = argparse.ArgumentParser(description="Evaluate span extraction performance")
    parser.add_argument("--gold_file", required=True, help="Path to gold standard JSONL file")
    parser.add_argument("--model_file", required=True, help="Path to model output JSONL file")
    parser.add_argument("--output_file", default="src/evaluation/qwen_no_chunk_evaluation.json",
                        help="Path to save detailed results (default: src/evaluation/qwen_no_chunk_evaluation.json)")
    parser.add_argument("--overlap_threshold", type=float, default=0.5,
                        help="Minimum overlap ratio for span matching (default: 0.5)")

    args = parser.parse_args()

    try:
        # Load data
        # Load data
        print(f"Loading gold spans from: {args.gold_file}")
        print(f"Loading model spans from: {args.model_file}")

        gold_sentence_dict = load_spans(args.gold_file)
        model_sentence_dict = load_spans(args.model_file)

        print(f"Loaded {len(gold_sentence_dict)} gold entries and {len(model_sentence_dict)} model entries")

        assert len(gold_sentence_dict) == len(model_sentence_dict), \
            f"Mismatch: {len(gold_sentence_dict)} gold vs {len(model_sentence_dict)} model entries"

        # At the moment the code is not using the text of the sentences
        all_gold_spans = dict()
        all_model_spans = dict()
        sentences = []

        for key, value in gold_sentence_dict.items():
            if key not in model_sentence_dict:
                print(f"Warning: Key {key} not found in model data")
                continue

            doc, sent_id = key

            sentences.append([_['text'] for _ in value])

            # Extract spans for each sentence, handling multiple instances per sentence
            gold_spans_list = [instance['spans'] for instance in value]
            model_spans_list = []
            for instance in model_sentence_dict[key]:
                llm_data = instance.get('llm', {})
                model_spans_list.extend(llm_data.get('accepted', []))
                model_spans_list.extend(llm_data.get('missing', []))

            # Flatten spans if we have multiple instances per sentence
            all_gold_spans[sent_id] = [span for spans in gold_spans_list for span in spans] if gold_spans_list else []
            all_model_spans[sent_id] = model_spans_list
        #

        # Run evaluation
        print(f"Running evaluation with overlap threshold: {args.overlap_threshold}")
        results = calculate_overlap_metrics(all_gold_spans, all_model_spans, sentences,
                                          overlap_threshold=args.overlap_threshold)

        # Print results
        print_evaluation_results(results)

        # Save detailed results
        with open(args.output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"\nDetailed results saved to: {args.output_file}")

    except FileNotFoundError as e:
        print(f"Error: File not found - {e}")
    except Exception as e:
        print(f"Error during evaluation: {e}")


if __name__ == "__main__":
    main()