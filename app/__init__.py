from flask import Flask, render_template, request
import pandas as pd
from engine.model import TrainLinearRegression, TrainDecisionTreeRegressor

app = Flask(__name__)

journals = pd.read_pickle('engine/journal_ids.pickle').sort_values('journal')
lr = TrainLinearRegression()
dtr = TrainDecisionTreeRegressor()


@app.route("/")
def index():
    preds = None
    journal_names = journals.to_dict('records')
    journal_names.insert(0, {'journal': 'None'})
    inp_journal = request.args.get('journal')
    inp_year = request.args.get('year')
    if inp_year and inp_year.isdigit():
        if str(inp_journal) != 'None':
            try:
                journal_id = journals.loc[journals['journal'] == inp_journal, 'JournalId'].values[0]
                print(journal_id)
                X = [[journal_id, int(inp_year)]]
                lr_pred = round(lr.predict(X)[0])
                dtr_pred = round(dtr.predict(X)[0])
                preds = [
                    {
                        "journal": inp_journal,
                        "year": inp_year,
                        "LR": lr_pred,
                        "DTR": dtr_pred
                    }
                ]
            except Exception as e:
                preds = None
        else:
            journ_copy = journals.copy()
            journ_copy['year'] = int(inp_year)
            inp_feat = journ_copy[['JournalId', 'year']].values
            journ_copy['LR'] = lr.predict(inp_feat)
            journ_copy['DTR'] = dtr.predict(inp_feat)
            preds = journ_copy.round(2).to_dict('records')
            print('')
    return render_template("index.html", journals=journal_names, preds=preds)
