import json
import math
import os
from collections import defaultdict
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


class draw_pic(object):
    def __init__(self):
        self.tmp_root = '../tmp_file/'
        self.dataset = '../Results/'
        self.results_root = '../Results/Imgs/'
        self.author_num_dic = self.read_dic('author_num_dic.json')
        self.if_dic = self.read_dic('jif_dic.json')
        self.pub_time_dic = self.read_dic('pub_time_dic.json')
        self.ref_age_median_dic = self.read_dic('ref_age_median_dic.json')
        self.ref_num_dic = self.read_dic('ref_num_dic.json')
        self.tfidf_median_dic = self.read_dic('cord_uid_tfidf_similarity.json')
        self.lda_median_dic = self.read_dic('cord_uid_lda_similarity.json')
        self.w2v_median_dic = self.read_dic('cord_uid_w2v_similarity.json')
        self.df = self.construct_dataframe()
        self.indicator_exchange = self.indicator_exchange()
        self.restrict_dic = self.get_restrict_dic()
        self.datetime = "_" + datetime.now().strftime("%H-%M-%S")

    def read_dic(self, json_name):
        with open(self.dataset + json_name, "r", encoding="utf-8") as fp:
            dic = json.load(fp)
        return dic

    def get_percentile(self, dictionary):
        dictionary_value = []
        for i in list(dictionary.values()):
            if i == i:
                dictionary_value.append(i)
        return np.percentile(dictionary_value, 95)

    def get_restrict_dic(self):
        restrict_dic = {}
        for indicator in ["author_num", "ref_age_median", "ref_num", "lda_median", "if", "tfidf_median", "w2v_median"]:
            dict_name = indicator + "_dic"
            dictionary = dict(eval("self." + dict_name))
            restrict_dic[indicator] = self.get_percentile(dictionary)
        return restrict_dic

    def construct_dataframe(self):
        cord_uid_list = list(self.pub_time_dic.keys())
        pub_time_list = [self.pub_time_dic.get(i, np.nan) for i in cord_uid_list]
        author_num_list = [self.author_num_dic.get(i, np.nan) for i in cord_uid_list]
        if_list = [self.if_dic.get(i, np.nan) for i in cord_uid_list]
        ref_age_median_list = [self.ref_age_median_dic.get(i, np.nan) for i in cord_uid_list]
        ref_num_list = [self.ref_num_dic.get(i, np.nan) for i in cord_uid_list]
        lda_median_list = [self.lda_median_dic.get(i, np.nan) for i in cord_uid_list]
        tfidf_median_list = [self.tfidf_median_dic.get(i, np.nan) for i in cord_uid_list]
        w2v_median_list = [self.w2v_median_dic.get(i, np.nan) for i in cord_uid_list]

        year_list = []
        month_list = []
        for t in pub_time_list:
            if len(t) == 10:
                year_list.append(t.split("-")[0])
                month_list.append(t.split("-")[1])
            elif len(t) == 4:
                year_list.append(t)
                month_list.append(np.nan)
            else:
                year_list.append(np.nan)
                month_list.append(np.nan)

        special_date_list = []
        for t in pub_time_list:
            if len(t) == 10:
                special_date_list.append(t[:7])
            else:
                special_date_list.append(np.nan)

        if_class_list = []
        for i in if_list:
            if not i == i:
                if_class_list.append(np.nan)
                continue
            if i <= 1:
                if_class_list.append("<1")
            elif i <= 5:
                if_class_list.append("1-5")
            elif i <= 10:
                if_class_list.append("5-10")
            elif i <= 15:
                if_class_list.append("10-15")
            elif i <= 20:
                if_class_list.append("15-20")
            elif i <= 25:
                if_class_list.append("20-25")
            elif i <= 30:
                if_class_list.append("25-30")
            elif i > 30:
                if_class_list.append(">30")
            else:
                print(i)

        df = pd.DataFrame({"cord_uid": cord_uid_list, "pub_time": pub_time_list,
                           "year": year_list, "month": month_list, "author_num": author_num_list,
                           "if_class": if_class_list, "if": if_list,
                           "ref_age_median": ref_age_median_list, "ref_num": ref_num_list,
                           "special_date": special_date_list, "tfidf_median": tfidf_median_list,
                           "w2v_median": w2v_median_list, "lda_median": lda_median_list})
        return df

    def draw_lineplot(self):
        year_dic = defaultdict(int)
        month_dic = defaultdict(int)
        for pub_time in self.pub_time_dic.values():
            if pub_time:
                year = pub_time.split("-")[0]
                if len(pub_time) == 10:
                    special_date = pub_time[:7]
                    if year in ("2019", "2020", "2021"):
                        month_dic[special_date] += 1
                year_dic[year] += 1

        year_tuple = sorted(list(year_dic.items()), key=lambda X: eval(X[0]))
        month_tuple = sorted(list(month_dic.items()), key=lambda X: (X[0].split("-")[0], X[0].split("-")[1]))

        year_list = [i[0] for i in year_tuple]
        new_year_list = []
        for i, j in enumerate(year_list):
            if i % 3 != 0:
                new_year_list.append("")
            else:
                new_year_list.append(j[2:])

        plt.figure(figsize=(8, 5))
        sns.lineplot(x=range(22), y=[math.log(i[1], 100) for i in year_tuple], marker="o", markersize=6, color="black")
        plt.grid(axis="y")
        plt.xticks(range(22), new_year_list)
        plt.xlabel("Year")
        plt.ylabel("log$_\mathrm{\mathregular{10^2}}$(Article number)")
        plt.savefig(self.results_root + 'count_year.png', bbox_inches='tight', dpi=200)
        plt.close()

        month_list = [i[0] for i in month_tuple]
        new_month_list = []
        for i, j in enumerate(month_list):
            if i % 5 != 0:
                new_month_list.append("")
            else:
                new_month_list.append(j[2:])

        plt.figure(figsize=(8, 5))
        sns.lineplot(x=range(36), y=[math.log(i[1], 100) for i in month_tuple], marker="o", markersize=6, color="black")
        plt.grid(axis="y")
        plt.xticks(range(36), new_month_list)
        plt.xlabel("Month")
        plt.ylabel("log$_\mathrm{\mathregular{10^2}}$(Article number)")
        plt.savefig(self.results_root + 'count_month.png', bbox_inches='tight', dpi=200)
        plt.close()

    def mkdir(self, dir_name):
        if not os.path.isdir(self.results_root + dir_name):
            os.mkdir(self.results_root + dir_name)

    def indicator_exchange(self):
        indicator_exchange = {"author_num": "Author Num",
                              "ref_age_median": "Ref Age",
                              "ref_num": "Ref Num",
                              "lda_median": "Topic Support",
                              "if": "JIF",
                              "tfidf_median": "Text Support",
                              "w2v_median": "Semantic Support"}
        return indicator_exchange

    def restrict_max(self, x, indicator):
        restrict_dic = self.restrict_dic
        if x > restrict_dic[indicator]:
            x = restrict_dic[indicator]
        if x < 0:
            x = 0
        return x

    def draw_heatmap_year(self):
        for indicator in ["author_num", "ref_age_median", "ref_num", "lda_median", "tfidf_median", "w2v_median"]:
            pt = self.df.pivot_table(index='if_class', columns='year', values=indicator, aggfunc=np.median)
            l = ['>30', '25-30', '20-25', '15-20', '10-15', '5-10', '1-5', '<1']
            pt["st"] = pt.index.astype('category')
            pt['st'].cat.reorder_categories(l, inplace=True)
            pt.sort_values('st', inplace=True)
            pt.set_index(["st"])
            del pt["st"]
            pt = pt.applymap(lambda X: self.restrict_max(X, indicator))
            pt.columns = [i[2:] for i in pt.columns]
            f, ax = plt.subplots(figsize=(20, 4), dpi=80)
            if indicator == "ref_num":
                fmt = ".0f"
            else:
                fmt = ".2g"
            sns.heatmap(pt, cmap="Blues", linewidths=0, ax=ax, annot=True, cbar=True, fmt=fmt)
            ax.set_xlabel('Year')
            ax.set_ylabel('Journal type')
            dir_name = indicator + self.datetime
            self.mkdir(dir_name)
            plt.savefig(self.results_root + '{}/2000_2021_{}_heatmap.png'.format(dir_name, indicator),
                        bbox_inches='tight', dpi=200)
            plt.close()

    def draw_heatmap_month(self):
        # 仅取2020,2021年
        df_2019_2020_2021 = self.df[(self.df["year"].isin(
            ["2019", "2020", "2021"]))]  # (df["special_date"]!="2020-01")&(df["special_date"]!="2020-02")
        for indicator in ["author_num", "ref_age_median", "ref_num", "lda_median", "tfidf_median", "w2v_median"]:
            pt = df_2019_2020_2021.pivot_table(index='if_class', columns='special_date', values=indicator,
                                               aggfunc=np.median)
            l = ['>30', '25-30', '20-25', '15-20', '10-15', '5-10', '1-5', '<1']
            pt["st"] = pt.index.astype('category')
            pt['st'].cat.reorder_categories(l, inplace=True)
            pt.sort_values('st', inplace=True)
            pt.set_index(["st"])
            del pt["st"]
            pt = pt.applymap(lambda X: self.restrict_max(X, indicator))
            pt.columns = [i[2:] for i in pt.columns]
            f, ax = plt.subplots(figsize=(22, 4), dpi=80)
            if indicator == "ref_num":
                fmt = ".0f"
            else:
                fmt = ".2f"
            if indicator in ("topic_median", "tfidf_median"):
                sns.heatmap(pt, cmap="Blues", linewidths=0, ax=ax, annot=True, mask=False, fmt=fmt,
                            annot_kws={'size': 11})  # weight':'bold' # annot_kws={'size':9, 'color':'black'},
            else:
                sns.heatmap(pt, cmap="Blues", linewidths=0, ax=ax, annot=True, mask=False, fmt=fmt)
            ax.set_xlabel('Month')
            ax.set_ylabel('Journal Type')
            dir_name = indicator + self.datetime
            self.mkdir(dir_name)
            plt.savefig(self.results_root + '{}/2019_2021_{}_heatmap.png'.format(dir_name, indicator),
                        bbox_inches='tight', dpi=200)

    def draw_boxplot_journal_year(self):
        df = self.df
        l = ['>30', '25-30', '20-25', '15-20', '10-15', '5-10', '1-5', '<1']
        for indicator in ["author_num", "ref_age_median", "ref_num", "lda_median", "tfidf_median", "w2v_median"]:
            fig = plt.figure(figsize=(4, 6), dpi=60)
            ax = sns.boxplot(y=df["if_class"], x=df[indicator], showfliers=False, order=l,
                             orient='h')  # ,order=old_list
            plt.xticks(rotation=90, fontsize=20)
            plt.grid(axis="x")
            plt.xlabel("")
            plt.ylabel("")
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['bottom'].set_visible(True)
            ax.spines['left'].set_visible(False)
            for patch in ax.artists:
                patch.set_facecolor((0, 0, 0, 0))
            dir_name = indicator + self.datetime
            self.mkdir(dir_name)
            plt.savefig(self.results_root + '{}/journals_class_{}_2000_2021.png'.format(dir_name, indicator),
                        bbox_inches='tight', dpi=200)
            plt.close()

    def draw_boxplot_journal_month(self):
        df = self.df
        df_2019_2020_2021 = df[(df["year"].isin(
            ["2019", "2020", "2021"]))]  # (df["special_date"]!="2020-01")&(df["special_date"]!="2020-02")
        l = ['>30', '25-30', '20-25', '15-20', '10-15', '5-10', '1-5', '<1']
        for indicator in ["author_num", "ref_age_median", "ref_num", "lda_median", "tfidf_median", "w2v_median"]:
            plt.figure(figsize=(4, 6), dpi=60)
            ax = sns.boxplot(y=df_2019_2020_2021["if_class"], x=df_2019_2020_2021[indicator], showfliers=False, order=l,
                             orient='h')  # ,order=old_list
            plt.xticks(rotation=90, fontsize=20)
            plt.ylabel("")
            plt.xlabel("")
            plt.grid(axis="x")
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['bottom'].set_visible(True)
            ax.spines['left'].set_visible(False)
            for patch in ax.artists:
                patch.set_facecolor((0, 0, 0, 0))
            dir_name = indicator + self.datetime
            self.mkdir(dir_name)
            plt.savefig(self.results_root + '{}/journals_class_{}_2019_2021.png'.format(dir_name, indicator),
                        bbox_inches='tight', dpi=200)
            plt.close()

    def draw_boxplot_year(self):
        df = self.df
        df_2000_2021 = df[df["year"].notna()]
        grouped_year_median = df_2000_2021.groupby(df_2000_2021["year"]).median()
        old_list = [str(i) for i in list(range(2000, 2022))]
        new_list = [str(i)[-2:] for i in old_list]
        max_length = len(new_list)
        for indicator in ["author_num", "if", "ref_age_median", "ref_num", "lda_median", "tfidf_median", "w2v_median"]:
            plt.figure(figsize=(12, 8), dpi=60)
            ax = sns.boxplot(x=df_2000_2021["year"], y=df_2000_2021[indicator], showfliers=False,
                             order=old_list)  # ,palette=day_pal
            sns.lineplot(data=grouped_year_median, x=grouped_year_median.index, y=indicator, color="black", linewidth=2,
                         marker="o", markersize=8)  # plt.get_cmap('Set2')(1)
            plt.xticks(range(max_length), new_list, rotation=90, fontsize=17)
            plt.xlabel("Year", fontsize=17)
            plt.ylabel(self.indicator_exchange[indicator], fontsize=17)
            plt.grid(axis="y")
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['bottom'].set_visible(True)
            ax.spines['left'].set_visible(True)
            num = 0
            for patch in ax.artists:
                num += 1
                if num in (4, 14, 21, 22):  # 标注03年，13年，20年（都是命名时间）
                    patch.set_facecolor((168 / 255, 206 / 255, 228 / 255, 1))
                else:
                    r, g, b, a = patch.get_facecolor()
                    patch.set_facecolor((r, g, b, 0))
            dir_name = indicator + self.datetime
            self.mkdir(dir_name)
            plt.savefig(self.results_root + '{}/2000_2021_{}_boxplot_blue.png'.format(dir_name, indicator),
                        bbox_inches='tight', dpi=200)
            plt.close()

    def draw_boxplot_month(self):
        df = self.df
        df_2019_2020_2021 = df[(df["year"].isin(["2019", "2020", "2021"])) & (df["special_date"].notna())]

        grouped_month_median = df_2019_2020_2021.groupby(df_2019_2020_2021["special_date"]).median()
        month_list = sorted(list(set(df_2019_2020_2021["special_date"])),
                            key=lambda x: (x.split("-")[0], x.split("-")[1]))
        old_list = [i[2:] for i in month_list]
        max_length = len(set(df_2019_2020_2021["special_date"]))
        new_list = []
        for i, j in enumerate(old_list):
            if i % 5 == 0:
                new_list.append(j)
            else:
                new_list.append("")
        for indicator in ["author_num", "if", "ref_age_median", "ref_num", "lda_median", "tfidf_median",
                          "w2v_median"]:  # "ref_num","ref_age_median","topic_median","author_num","if"
            plt.figure(figsize=(12, 8), dpi=60)
            ax = sns.boxplot(x=df_2019_2020_2021["special_date"], y=df_2019_2020_2021[indicator], showfliers=False,
                             order=month_list)
            sns.lineplot(data=grouped_month_median, x=grouped_month_median.index, y=indicator, color="black",
                         linewidth=2, marker="o", markersize=8)  # label="median",
            plt.xticks(range(max_length), new_list, rotation=90, fontsize=17)
            plt.grid(axis="y")
            plt.xlabel("Month", fontsize=17)
            plt.ylabel(self.indicator_exchange[indicator], fontsize=17)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['bottom'].set_visible(True)
            ax.spines['left'].set_visible(True)
            num = 0
            for patch in ax.artists:
                num += 1
                if num >= 13 and num < 25:
                    patch.set_facecolor((168 / 255, 206 / 255, 228 / 255, 0.5))
                elif num >= 25:
                    patch.set_facecolor((168 / 255, 206 / 255, 228 / 255, 1))
                else:
                    r, g, b, a = patch.get_facecolor()
                    patch.set_facecolor((r, g, b, 0))
            dir_name = indicator + self.datetime
            self.mkdir(dir_name)
            plt.savefig(self.results_root + '{}/2019_2021_{}_boxplot_blue.png'.format(dir_name, indicator),
                        bbox_inches='tight', dpi=200)
            plt.close()

    def draw_relation(self):
        df = self.df
        indicator_exchange = self.indicator_exchange
        num = 0
        for year_scope in [["2019"], ["2020"], ["2021"]]:
            num += 1
            df_select_year = df[df["year"].isin(year_scope)]
            corr_df = df_select_year[
                ["w2v_median", "lda_median", "tfidf_median", "ref_age_median", "ref_num", "author_num", "if"]]
            corr_df.columns = [indicator_exchange[i] for i in
                               ["w2v_median", "lda_median", "tfidf_median", "ref_age_median", "ref_num",
                                "author_num", "if"]]
            mask = np.zeros_like(corr_df.corr(), dtype=np.bool)
            mask[np.triu_indices_from(mask)] = True
            plt.subplots(figsize=(8, 8))
            sns.diverging_palette(240, 10, n=100)
            sns.heatmap(corr_df.corr(), square=True, annot=True, cmap="vlag", center=0, mask=mask, vmin=-0.2,
                        vmax=0.2)  # ,norm=norm
            dir_name = "relation" + self.datetime
            self.mkdir(dir_name)
            plt.savefig(self.results_root + '{}/relation_{}.png'.format(dir_name, year_scope[0]), bbox_inches='tight',
                        dpi=200)
            plt.close()


if __name__ == '__main__':
    params = {'font.family': 'serif',
              'font.serif': 'Times New Roman',
              'font.style': 'normal',
              'font.weight': 'normal',  # or 'blod'
              'font.size': 13,  # or large,small'medium'
              }
    plt.rcParams.update(params)
    print("init...")
    draw_pic = draw_pic()
    draw_pic.draw_lineplot()
    print("done lineplot...")
    draw_pic.draw_heatmap_year()
    print("done heatmap_year...")
    draw_pic.draw_heatmap_month()
    print("done heatmap_month...")
    draw_pic.draw_boxplot_journal_year()
    print("done boxplot_journal_year...")
    draw_pic.draw_boxplot_journal_month()
    print("done boxplot_journal_month...")
    draw_pic.draw_boxplot_year()
    print("done boxplot_year...")
    draw_pic.draw_boxplot_month()
    print("done boxplot_month...")
    draw_pic.draw_relation()
    print("done relation...")
