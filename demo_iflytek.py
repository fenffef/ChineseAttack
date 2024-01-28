import ssl
ssl._create_default_https_context = ssl._create_unverified_context
import OpenAttack
import datasets
import transformers
import argparse
import os

def dataset_mapping(x):
    return {
        "x": x["sentence"],
        "y": x["label"],
    }


def main(args):
    print("New Attacker")
    if args.attack == "textbugger":
        attacker = OpenAttack.attackers.TextBuggerAttacker(lang=args.language)
    elif args.attack == "pwws":
        attacker = OpenAttack.attackers.TextBuggerAttacker(lang=args.language)


    print("Load model")
    tokenizer = transformers.AutoTokenizer.from_pretrained(args.model_name)
    model = transformers.AutoModelForSequenceClassification.from_pretrained(args.model_name, num_labels=120, output_hidden_states=False, ignore_mismatched_sizes=True)
    clsf = OpenAttack.classifiers.TransformersClassifier(model, tokenizer, model.bert.embeddings.word_embeddings)

    print("Loading dataset")
    dataset = datasets.load_dataset(args.dataset_name, split=args.split).map(function=dataset_mapping)

    print("Start attack")
    attack_eval = OpenAttack.AttackEval(attacker, clsf, metrics=[
        OpenAttack.metric.Fluency(),
        OpenAttack.metric.GrammaticalErrors(),
        OpenAttack.metric.EditDistance(),
        OpenAttack.metric.ModificationRate()
    ])
    results = attack_eval.eval(dataset, visualize=True, progress_bar=True)

    # 根据任务进行输出文件命名
    output_file = f"{args.dataset_name}_{args.model_name}_{args.attack}_{args.split}.txt"
    
    if not os.path.exists(output_file):
    # 如果路径不存在，创建目录结构
        os.makedirs(os.path.dirname(output_file), exist_ok=True)

    with open(output_file, 'w') as file:
        for key, value in results.items():
            file.write(f"{key}: {value}\n")



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test an attacker on a dataset")
    parser.add_argument("--model_name", type=str, default="fenffef/bert-freelb-tnews", help="Pretrained model name or path")
    parser.add_argument("--language", type=str, default="chinese", help="Language of the attacker")
    parser.add_argument("--dataset_name", type=str, default="fenffef/tnews", help="Name of the dataset")
    parser.add_argument("--split", type=str, default="validation[:]", help="Split of the dataset")
    parser.add_argument("--attack", type=str, default="textbugger", help="Name of the dataset")
    args = parser.parse_args()
    main(args)