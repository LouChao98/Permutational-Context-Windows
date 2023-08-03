from pathlib import Path
import numpy as np
import pandas as pd

result = []
for opath in Path('logs').glob('*/*.npy'):
    m = np.load(opath).mean()
    result.append({'path': str(opath).strip('logs/'), 'result': m})

df = pd.DataFrame(result)
df = df.sort_values(by='path')
pd.options.display.max_colwidth = 100
print(df)