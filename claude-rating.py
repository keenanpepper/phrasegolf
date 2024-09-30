import os
from anthropic import Anthropic
from dotenv import load_dotenv
from typing import List, Tuple
import re

load_dotenv()

client = Anthropic(
    api_key = os.environ.get("ANTHROPIC_API_KEY")
)

def evaluate_idioms(idioms: List[str]) -> List[Tuple[str,int]]:
    idioms_str = "".join(idioms)

    reg = re.compile("<idiom>(.*)</idiom><score>(.*)</score>")

    response = client.messages.create(
        max_tokens=1024,
        messages=[
            {
                "role": "user",
                "content": f"Please rank all the following idioms from 1 meaning very obscure and unlikely to be known to 10 meaning universally known and understood by all fluent English speakers. Use the format \"<idiom>after all</idiom><score>10</score>\" and keep the idioms in the same order they were given, to make sure none are left out:\n\n<idioms>{idioms_str}</idioms>",
            }
        ],
        model="claude-3-sonnet-20240229",
    )

    for block in response.content:
        raw_pairs = reg.findall(block.text)
        cooked_pairs = [(pair[0], int(pair[1])) for pair in raw_pairs if pair[0] != "after all"]
        # note to self: make sure "after all" is manually added after all is said and done
        return cooked_pairs

def batch_reader(file_path, batch_size=20):
    with open(file_path, 'r') as file:
        while True:
            batch = [next(file, None) for _ in range(batch_size)]
            batch = [line for line in batch if line is not None]
            if not batch:
                break
            yield batch

BATCH_SIZE = 20
scored = []
with open("scored-by-claude.tsv", "w") as scored_file:
    for batch in batch_reader("targets.tsv", BATCH_SIZE):
        scored += evaluate_idioms(batch)
        print(f"{len(scored)} scored")
    for pair in scored:
        scored_file.write(pair[0] + "\t" + str(pair[1]) + "\n")
