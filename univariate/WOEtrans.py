import configparser

import numpy as np
import pandas as pd

from univariate.common.woe import WoE
import matplotlib.pyplot as plt
import os


class WOEtrans(object):
    def __init__(self, property_dir):
        self.config = configparser.ConfigParser()
        self.config.optionxform = str
        self.config.read(property_dir)

        self.data_dir = self.config['DIR']['data_dir']
        self.iv_dir = self.config['DIR']['iv_dir']
        self.woe_data_dir = self.config['DIR']['woe_data_dir']
        self.fig_dir = self.config['DIR']['fig_dir']

        self.data_orig = pd.read_csv(self.data_dir, index_col=0, encoding='utf8')
        self.data = self.resample(self.data_orig)

        self.id_vars = ['bbd_qyxx_id', 'company_name', 'label']
        # self.id_vars = ['bbd_qyxx_id', 'label']
        self.all_vars = list(set(self.data.columns) - set(self.id_vars))
        self.category_vars = ['company_companytype', 'company_county', 'ipo_company', 'feature_5_c_i', 'feature_5_p_i',
                              'company_industry']
        self.numeric_vars = list(set(self.all_vars) - set(self.category_vars))

        self.trans_method = {}
        self.iv = pd.DataFrame(columns=['indicator', 'iv'])

    # 重抽样
    def resample(self, data, black_ratio=0.2):
        black = data[data['label'] == 1].reset_index(drop=True)
        white = data[data['label'] == 0].reset_index(drop=True)
        black_num = len(black)
        white_num = len(white)
        new_black_num = round(white_num * black_ratio)
        np.random.seed(42)
        new_black_indexes = np.random.randint(black_num, size=new_black_num - black_num)
        new_black = black.append(black.iloc[new_black_indexes]).reset_index(drop=True)
        new_data = white.append(new_black).reset_index(drop=True)
        return new_data

    # 训练WOE
    def cal_trans(self):
        for _index, i in enumerate(self.all_vars):
            # print(_index, i)
            if i in self.numeric_vars:
                if self.data[i].unique().size < 8:
                    woe_tmp = WoE(v_type='d', t_type='b')
                    woe_tmp.fit(self.data[i], self.data['label'])
                else:
                    try:
                        woe_tmp = WoE(qnt_num=3, v_type='c', t_type='b')
                        woe_tmp.fit(self.data[i], self.data['label'])
                    except ValueError:
                        try:
                            woe_tmp = WoE(qnt_num=2, spec_values=[self.data[i].value_counts().index[0]], v_type='c',
                                          t_type='b')
                            woe_tmp.fit(self.data[i], self.data['label'])

                        except ValueError:
                            try:
                                woe_tmp = WoE(qnt_num=1, spec_values=[self.data[i].value_counts().index[0],
                                                                      self.data[i].value_counts().index[1]], v_type='c',
                                              t_type='b')
                                woe_tmp.fit(self.data[i], self.data['label'])

                            except ValueError:
                                try:
                                    woe_tmp = WoE(qnt_num=1, spec_values=[self.data[i].value_counts().index[0],
                                                                          self.data[i].value_counts().index[1],
                                                                          self.data[i].value_counts().index[2]], v_type='c',
                                                  t_type='b')
                                    woe_tmp.fit(self.data[i], self.data['label'])

                                except ValueError:
                                    try:
                                        woe_tmp = WoE(qnt_num=1, spec_values=[self.data[i].value_counts().index[0],
                                                                              self.data[i].value_counts().index[1],
                                                                              self.data[i].value_counts().index[2],
                                                                              self.data[i].value_counts().index[3]],
                                                      v_type='c', t_type='b')
                                        woe_tmp.fit(self.data[i], self.data['label'])

                                    except ValueError:
                                        woe_tmp = WoE(qnt_num=1, spec_values=[self.data[i].value_counts().index[0],
                                                                              self.data[i].value_counts().index[1],
                                                                              self.data[i].value_counts().index[2],
                                                                              self.data[i].value_counts().index[3],
                                                                              self.data[i].value_counts().index[4]],
                                                      v_type='c', t_type='b')
                                        woe_tmp.fit(self.data[i], self.data['label'])

            else:
                woe_tmp = WoE(v_type='d', t_type='b')
                woe_tmp.fit(self.data[i], self.data['label'])

            self.trans_method[i] = woe_tmp
            # 保存iv
            self.iv = self.iv.append(pd.Series({'indicator': i, 'iv': woe_tmp.iv}), ignore_index=True)
        self.iv.index.name = 'index'
        self.iv.to_csv(self.iv_dir, encoding='utf8')

    def do_trans(self):
        new_data = self.data_orig
        new_data_after_trans = pd.DataFrame()
        for i in self.all_vars:
            new_data_after_trans[i] = self.trans_method[i].transform(new_data[i])['woe']
        for j in self.id_vars:
            new_data_after_trans[j] = new_data[j]
        new_data_after_trans.to_csv(self.woe_data_dir, encoding='utf8')

    def plot(self):
        for i in self.all_vars:
            if all(self.data[i].isnull()):
                continue
            try:
                self.trans_method[i].plot()
                plt.title('{}_woe_plot'.format(i))
                plt.text(0.5, 0.9, 'iv: {}'.format(round(self.trans_method[i].iv, 3)), ha='center', va='center', transform=plt.gca().transAxes)
                plt.savefig(os.path.join(self.fig_dir, '{}_woe_plot.jpg'.format(i)))
                plt.close('all')
            except ValueError as e:
                print(i, e)


if __name__ == '__main__':
    woe1 = WOEtrans(property_dir='data/property.ini')
    woe1.cal_trans()
    woe1.do_trans()
    woe1.plot()
