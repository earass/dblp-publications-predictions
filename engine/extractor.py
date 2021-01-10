import pandas as pd
from engine.plots import bar_plot, line_plot
from engine.my_parser import parse_xml

pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)


class DBLPExtractor:
    def __init__(self, parse_data=False):
        self.df = DBLPExtractor.read_data(parse_data)
        self.journal_ids = pd.DataFrame()

    @classmethod
    def read_data(cls, parse_data):
        if parse_data:
            parse_xml()
        path = 'data/articles_2.csv'
        return pd.read_csv(path)

    def preprocess(self, export_csv):
        self.df['year'] = self.df['year'].astype('Int64')

        # exclude 2020
        self.df = self.df.loc[self.df['year'] < 2020]

        # journal numeric ids
        self.journal_ids = self.df['journal'].drop_duplicates().reset_index()
        self.journal_ids['JournalId'] = list(range(1, len(self.journal_ids) + 1))
        self.journal_ids.to_pickle('journal_ids.pickle')

        # Publications by journals per year
        journal_vol_per_year = self.df.groupby(['journal', 'year']).agg(
            count=('author', 'count')
        ).sort_values('count', ascending=False).reset_index()
        journal_vol_per_year = journal_vol_per_year.merge(self.journal_ids[['JournalId', 'journal']], on='journal')
        print(journal_vol_per_year)

        if export_csv:
            # exporting as csv
            journal_vol_per_year.to_csv('data/journal_vol_per_year.csv', index=False)

    def exploration(self):
        # total records
        print("Total records: ", len(self.df))

        # missing year
        missing_year_cond = self.df['year'].isnull()
        print("Missing year: ", len(self.df[missing_year_cond]))

        # missing journal
        missing_journal_cond = self.df['journal'].isnull()
        print("Missing journal: ", len(self.df[missing_journal_cond]))

        # missing both journal and year
        print("Missing both journal and year: ", len(self.df[(missing_journal_cond) & (missing_year_cond)]))

        # yearly volume
        year_vol = self.df['year'].value_counts()
        bar_plot(x=year_vol.index, y=year_vol.values, xlabel='Year', ylabel='Number of publications',
                 title='Number of publications by Year')

        # Publications by top 15 journals
        journal_vol = self.df['journal'].value_counts().head(15)
        bar_plot(x=journal_vol.index, y=journal_vol.values, xlabel='Journal', ylabel='Number of publications',
                 title='Number of publications by top 15 journals', degrees=90)
        print("Top 15 Journals")
        print(journal_vol)

        # Publications by top 15 journals by year
        top_15_journals = list(journal_vol.index)
        top_15_journal_vol_per_year = self.df[self.df['journal'].isin(top_15_journals)].groupby(
            ['journal', 'year']).agg(
            count=('author', 'count')
        ).sort_values('count', ascending=False).reset_index().pivot(index='year', columns='journal', values='count')
        line_plot(top_15_journal_vol_per_year, 'Top 15 Journal volume by Year')

    def execute(self):
        # self.preprocess(export_csv=False)
        self.exploration()


if __name__ == '__main__':
    ext = DBLPExtractor()
    ext.execute()
