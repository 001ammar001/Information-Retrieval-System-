import argparse
import numpy as np
from collections import defaultdict
from typing import Dict, List, Tuple

import ir_datasets
from app.online.tfidf_service import tfidf_service
from app.online.bert_service import bert_service
from app.online.hybrid_service import hybrid_service
from app.online.topic_detection_service import topic_detection_service
from app.services.text_processing_services import text_processing_service

class ModelEvaluator:
    def get_qrels(self, dataset) -> Tuple[Dict[str, Dict[str, int]], Dict[str, List[str]]]:
        qrel_dict = defaultdict(dict)
        real_relevant = defaultdict(list)
        for qrel in dataset.qrels_iter():
            query_id = qrel[0]
            doc_id = qrel[1]
            relevance = qrel[2]
            qrel_dict[query_id][doc_id] = relevance
            if relevance > 0:
                real_relevant[query_id].append(doc_id)
        return dict(qrel_dict), dict(real_relevant)

    def get_retrieved_docs(self, dataset, dataset_name: str, model_type: str, top_k: int = 10) -> Dict[str, List[str]]:
        retrieved_docs = {}
        queries_count = len(list(dataset.queries_iter()))
        print(f"Processing {queries_count} queries...")
        for i, query in enumerate(dataset.queries_iter()):
            if i % 100 == 0 or i == queries_count - 1:
                print(f"Processed {i} queries...")
            processed_query = text_processing_service.process_text(query.text)
            if model_type == "tfidf":
                doc_ids = [res["document_id"] for res in tfidf_service.find_similar_documents(query.text, processed_query, dataset_name, top_k)]
            elif model_type == "bert":
                doc_ids = [res["document_id"] for res in bert_service.find_similar_documents(processed_query, dataset_name, top_k)]
            elif model_type == "hybrid":
                doc_ids = [res["document_id"] for res in hybrid_service.find_similar_documents(query.text, processed_query, dataset_name, top_k)]
            elif model_type =="topic":
                doc_ids=[res["document_id"] for res in topic_detection_service.find_similar_documents(query.text, processed_query, dataset_name, top_k)]
            else:
                raise ValueError(f"Unknown model type: {model_type}")
            retrieved_docs[query.query_id] = doc_ids
        return retrieved_docs

    def calculate_average_precision(self, ranked_list: List[str], true_relevant_docs: set, top_k: int) -> float:
        if not true_relevant_docs: return 0.0

        relevant_count, precision_sum = 0, 0.0
        
        for i, doc_id in enumerate(ranked_list):
            if i >= top_k: break
            if doc_id in true_relevant_docs:
                relevant_count += 1
                precision_at_k = relevant_count / (i + 1)
                precision_sum += precision_at_k
        
        if relevant_count == 0: return 0.0

        return precision_sum / relevant_count

    def calculate_map(self, qrel_dict: Dict[str, Dict[str, int]], retrieved_docs: Dict[str, List[str]], top_k: int) -> float:
        average_precisions = []
        
        for query_id, retrieved in retrieved_docs.items():
            if query_id not in qrel_dict: continue
                
            query_qrels = qrel_dict[query_id]
            true_relevant_docs = {
                doc_id for doc_id, relevance in query_qrels.items() 
                if relevance > 0
            }
            
            ap = self.calculate_average_precision(retrieved, true_relevant_docs, top_k)
            if ap > 0: average_precisions.append(ap)
        
        if not average_precisions: return 0.0
            
        return np.mean(average_precisions) * 100

    def calculate_mrr(self, qrel_dict: Dict[str, Dict[str, int]], retrieved_docs: Dict[str, List[str]]) -> float:
        reciprocal_ranks = []
        for query_id, retrieved in retrieved_docs.items():
            if query_id not in qrel_dict:
                continue
            for i, doc_id in enumerate(retrieved):
                if doc_id in qrel_dict[query_id] and qrel_dict[query_id][doc_id] > 0:
                    reciprocal_ranks.append(1 / (i + 1))
                    break
        if not reciprocal_ranks:
            return 0.0
        return np.mean(reciprocal_ranks) * 100

    def calculate_recall_at_k(self, real_relevant: Dict[str, List[str]], retrieved_docs: Dict[str, List[str]]) -> float:
        recall_scores = []
        for query_id, relevant in real_relevant.items():
            if query_id not in retrieved_docs or not relevant: continue
            retrieved = retrieved_docs[query_id]
            relevant_retrieved = [doc_id for doc_id in retrieved if doc_id in relevant]
            recall = len(relevant_retrieved) / len(relevant)
            recall_scores.append(recall)
        if not recall_scores:
            return 0.0
        return np.mean(recall_scores) * 100

    def calculate_precision_at_k(self, qrel_dict: Dict[str, Dict[str, int]], retrieved_docs: Dict[str, List[str]], k: int = 10) -> float:
        precision_scores = []
        for query_id, retrieved in retrieved_docs.items():
            if query_id not in qrel_dict:
                continue
            relevant_retrieved = [doc_id for doc_id in retrieved[:k] if doc_id in qrel_dict[query_id] and qrel_dict[query_id][doc_id] > 0]
            precision = len(relevant_retrieved) / k
            precision_scores.append(precision)
        if not precision_scores:
            return 0.0
        return np.mean(precision_scores) * 100

    def calculate_ndcg(self, qrel_dict: Dict[str, Dict[str, int]], retrieved_docs: Dict[str, List[str]], k: int = 10) -> float:
        ndcg_scores = []
        for query_id, retrieved in retrieved_docs.items():
            if query_id not in qrel_dict:
                continue

            query_qrels = qrel_dict[query_id]
            
            # Calculate DCG
            dcg = 0.0
            for i, doc_id in enumerate(retrieved[:k]):
                relevance = query_qrels.get(doc_id, 0)
                if i == 0: 
                    dcg += (2 ** relevance - 1)
                else:
                    dcg += (2 ** relevance - 1) / np.log2(i + 1)

            # Calculate IDCG
            ideal_relevances = sorted([relevance for relevance in query_qrels.values() if relevance > 0], reverse=True)
            idcg = 0.0
            for i, relevance in enumerate(ideal_relevances[:k]):
                if i == 0:
                    idcg += relevance
                else:
                    idcg += relevance / np.log2(i + 1)

            if idcg > 0:
                ndcg_scores.append(dcg / idcg)
            else:
                ndcg_scores.append(0.0) # If no relevant documents, NDCG is 0

        if not ndcg_scores: return 0.0
        
        return np.mean(ndcg_scores) * 100


    def evaluate(self, qrels_dict: Dict[str, Dict[str, int]], relevant_map: Dict[str, List[str]], retrieved_docs: Dict[str, List[str]], top_k) -> Dict[str, float]:
        print("Calculating evaluation metrics...")
        map_score = self.calculate_map(qrels_dict, retrieved_docs, top_k)
        mrr_score = self.calculate_mrr(qrels_dict, retrieved_docs)
        mean_recall = self.calculate_recall_at_k(relevant_map, retrieved_docs)
        precision_at_k = self.calculate_precision_at_k(qrels_dict, retrieved_docs, k=top_k)
        ndcg_at_k = self.calculate_ndcg(qrels_dict, retrieved_docs, k=top_k) # Calculate NDCG
        results = {
            "@K": top_k,
            "MAP": map_score,
            "MRR": mrr_score,
            "Recall@K": mean_recall,
            "Precision@K": precision_at_k,
            "NDCG@K": ndcg_at_k # Add NDCG to results
        }
        return results

    def evaluate_model(self, dataset_name: str, model_type: str, top_k: int = 10) -> Dict[str, float]:
        print(f"\n{'='*60}")
        print(f"EVALUATING {model_type.upper()} MODEL ON {dataset_name.upper()} DATASET")
        print(f"{'='*60}")
        # Load dataset
        if dataset_name.lower() == "antique":
            dataset = ir_datasets.load('antique/test/non-offensive')
        elif dataset_name.lower() == "quora":
            dataset = ir_datasets.load('beir/quora/test')
        else:
            raise ValueError(f"Unsupported dataset: {dataset_name}")
        # Get qrels
        print("Extracting query relevance judgments...")
        qrels_dict, relevant_map = self.get_qrels(dataset)
        # Get retrieved documents
        print(f"Retrieving documents using {model_type} model...")
        retrieved_docs = self.get_retrieved_docs(dataset, dataset_name, model_type, top_k)
        # Calculate metrics
        results = self.evaluate(qrels_dict, relevant_map, retrieved_docs, top_k)
        # Print results
        print(f"\n{'='*40} RESULTS {'='*40}")
        print(f"MAP: {results['MAP']:.2f}%")
        print(f"MRR: {results['MRR']:.2f}%")
        print(f"Recall@{top_k}: {results['Recall@K']:.2f}%")
        print(f"Precision@{top_k}: {results['Precision@K']:.2f}%")
        print(f"NDCG@{top_k}: {results['NDCG@K']:.2f}%")
        print(f"{'='*50}")
        return results


evaluator = ModelEvaluator()

def main():
    parser = argparse.ArgumentParser(description='Evaluate IR models on datasets')
    parser.add_argument('--dataset', type=str, default='antique', 
                        choices=['antique', 'quora'], 
                        help='Dataset to evaluate on (default: antique)')
    parser.add_argument('--model', type=str, default='tfidf', 
                        choices=['tfidf', 'bert','topic', 'hybrid', 'both'], 
                        help='Model type to evaluate (default: tfidf)')
    parser.add_argument('--top_k', type=int, default=10, 
                        help='Number of top documents to retrieve (default: 10)')
    args = parser.parse_args()

    if args.model in ['tfidf', 'both']:
        try:
            evaluator.evaluate_model(args.dataset, 'tfidf', args.top_k)
        except Exception as e:
            print(f"Error evaluating TF-IDF model: {e}")
    if args.model in ['bert', 'both']:
        try:
            evaluator.evaluate_model(args.dataset, 'bert', args.top_k)
        except Exception as e:
            print(f"Error evaluating BERT model: {e}")
    if args.model in ['hybrid', 'both']:
        try:
            evaluator.evaluate_model(args.dataset, 'hybrid', args.top_k)
        except Exception as e:
            print(f"Error evaluating Hybrid model: {e}")
    if args.model in ['topic']:
        try:
            evaluator.evaluate_model(args.dataset,'topic',args.top_k)
        except Exception as e:
            print(f"Error evaluating topic model: {e}")

if __name__ == "__main__":
    main()
# ======================================== RESULTS ========================================
# MAP: 70.72%
# MRR: 75.37%
# Mean Precision: 10.26%
# Mean Recall: 76.52%
# Precision@10: 10.26%
# ==================================================

# ======================================== RESULTS ======================================== service khaled 0.5 0.5
# MAP: 81.00%
# MRR: 86.08%
# Mean Precision: 10.95%
# Mean Recall: 81.76%
# Precision@10: 10.93%
# ==================================================

# ======================================== RESULTS ========================================
# MAP: 80.44%
# MRR: 85.20%
# Mean Precision: 11.03%
# Mean Recall: 81.28%
# Precision@10: 11.03%
# ==================================================

def calculate_precision_at_k(self, retrieved: List[str], query_docs: set, top_k: int): 
    if not query_docs:
        return 0.0

    relevant_count = 0
    
    for i, doc_id in enumerate(retrieved):
        if i >= top_k:
            break
        if doc_id in query_docs:
            relevant_count += 1
    
    if relevant_count == 0:
        return 0.0

    return relevant_count / min(len(retrieved), top_k)