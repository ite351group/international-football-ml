"""Preview-only validation experiments. Existing final-test reports stay untouched."""
import sys, json, platform, hashlib, argparse
from pathlib import Path
from datetime import datetime, timezone
import pandas as pd
import numpy as np
import sklearn
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.dummy import DummyClassifier
from sklearn.metrics import accuracy_score, f1_score, balanced_accuracy_score, recall_score, log_loss

ROOT = Path(__file__).resolve().parent.parent
parser = argparse.ArgumentParser()
parser.add_argument('--repo',type=Path,default=ROOT / 'work/group_repo/international-football-ml')
parser.add_argument('--output',type=Path,default=ROOT / 'outputs/defence_preview')
args = parser.parse_args()
REPO = args.repo.resolve()
OUT = args.output.resolve()
sys.path.insert(0, str(REPO / 'src'))
import train


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    raw = pd.read_csv(REPO / 'data/results.csv')
    raw['source_row'] = range(len(raw))
    raw['date'] = pd.to_datetime(raw['date'], errors='coerce')
    for col in ['home_score','away_score']: raw[col] = pd.to_numeric(raw[col], errors='coerce')
    neutral = raw.neutral.astype(str).str.lower().map({'true':True, 'false':False})
    score_bad = ((raw[['home_score','away_score']] < 0) | (raw[['home_score','away_score']] % 1 != 0)).any(axis=1)
    audit = {'rows':len(raw), 'missing_by_column':raw.isna().sum().to_dict(), 'exact_duplicates_excluding_row_id':int(raw.drop(columns='source_row').duplicated().sum()), 'invalid_neutral':int(neutral.isna().sum()), 'invalid_score_rows':int(score_bad.sum()), 'same_team_rows':int((raw.home_team == raw.away_team).sum()), 'fixture_key_duplicates':int(raw.duplicated(['date','home_team','away_team']).sum()), 'max_goals_single_team':float(raw[['home_score','away_score']].max().max())}
    raw['neutral'] = neutral
    clean = raw.dropna(subset=['date','home_team','away_team','home_score','away_score','tournament','neutral']).loc[~score_bad].copy()
    clean[['home_score','away_score']] = clean[['home_score','away_score']].astype(int)
    # Do not construct features from the already inspected final holdout.
    clean = clean[clean.date < '2019-11-19']
    print('Building pre-holdout features', flush=True)
    features = train.build_features(clean)
    features['match_type'] = np.where(features.tournament == 'Friendly','Friendly','Other tournament')
    features = features[features.date >= '1990-01-01'].copy()
    context = ['neutral','match_type']
    recent = ['home_form5','away_form5','home_form10','away_form10','form5_diff','form10_diff','home_rest_days','away_rest_days','rest_diff']
    elo = ['home_elo','away_elo','elo_diff','elo_home_expectancy']
    groups = {'context':([],context), 'context_form':(recent,context), 'context_form_elo':(recent+elo,context), 'full_features':(train.NUMERIC,train.CATEGORICAL)}
    folds = [('2010-01-01','2013-01-01'),('2013-01-01','2016-01-01'),('2016-01-01','2019-11-19')]
    rows=[]
    settings=[('context','none',1),('context_form','none',1),('context_form_elo','none',1),('full_features','none',1),('context_form_elo','balanced',1),('context_form_elo','balanced',.1),('context_form_elo','balanced',10)]
    for start,end in folds:
        tr,va = features[features.date < start],features[(features.date >= start)&(features.date < end)]
        assert tr.date.max() < va.date.min() and va.date.max() < pd.Timestamp('2019-11-19')
        for group,weight,c in [('majority','none',1)]+settings:
            print(start, group, weight, c, flush=True)
            if group == 'majority':
                model=DummyClassifier(strategy='most_frequent').fit(np.zeros((len(tr),1)),tr.target)
                pred=model.predict(np.zeros((len(va),1))); prob=model.predict_proba(np.zeros((len(va),1)))
            else:
                nums,cats=groups[group]
                blocks=[('cat',OneHotEncoder(handle_unknown='ignore'),cats)]
                if nums: blocks.append(('num',Pipeline([('impute',SimpleImputer(strategy='median')),('scale',StandardScaler())]),nums))
                model=Pipeline([('preprocess',ColumnTransformer(blocks)),('model',LogisticRegression(C=c,class_weight=None if weight=='none' else weight,max_iter=3000,random_state=42))]).fit(tr, tr.target)
                pred=model.predict(va);prob=model.predict_proba(va)
            rows.append({'validation_start':start,'validation_end_exclusive':end,'train_rows':len(tr),'validation_rows':len(va),'feature_set':group,'class_weight':weight,'C':c,'accuracy':accuracy_score(va.target,pred),'balanced_accuracy':balanced_accuracy_score(va.target,pred),'macro_f1':f1_score(va.target,pred,average='macro'),'draw_recall':recall_score(va.target,pred,labels=['draw'],average=None,zero_division=0)[0],'log_loss':log_loss(va.target,prob,labels=model.classes_)})
    results=pd.DataFrame(rows)
    results.to_csv(OUT/'validation_results.csv',index=False)
    summary=results.groupby(['feature_set','class_weight','C'])[['accuracy','balanced_accuracy','macro_f1','draw_recall','log_loss']].agg(['mean','std'])
    summary.to_csv(OUT/'validation_summary.csv')
    meta={'run_utc':datetime.now(timezone.utc).isoformat(),'python':platform.python_version(),'pandas':pd.__version__,'sklearn':sklearn.__version__,'input_sha256':hashlib.sha256((REPO/'data/results.csv').read_bytes()).hexdigest(),'audit':audit,'folds':folds,'feature_groups':groups,'scope':'Local preview. No final holdout retraining/evaluation. These past windows were part of an earlier broad historical experiment and are not pristine blind data. No confidence/significance claim. Mean across three windows is unweighted.'}
    (OUT/'experiment_metadata.json').write_text(json.dumps(meta,indent=2),encoding='utf-8')
    print(summary.to_string(),flush=True)

if __name__=='__main__':main()
