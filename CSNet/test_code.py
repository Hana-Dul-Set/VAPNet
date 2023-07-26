from tqdm import tqdm

a = [x for x in range(0, 100000)]
for index, data in tqdm(enumerate(a), total=len(a)):
    data += 1