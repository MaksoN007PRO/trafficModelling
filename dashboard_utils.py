import pandas as pd
import numpy as np
import streamlit as st
import pickle
from xgboost import XGBClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_selection import SelectPercentile
import seaborn as sns
from matplotlib.patches import Rectangle
import matplotlib.pyplot as plt
import lime
import lime.lime_tabular

def read_array(file):
    content = np.loadtxt(file, delimiter=",")
    return content




def load_xgb_pipeline(preprocessor):

    xgb_pipeline = Pipeline([('preprocessor', preprocessor),
                             ('select', SelectPercentile(percentile = 33)),
                             ('model', XGBClassifier(random_state = 2))])
    return xgb_pipeline



def load_xgb_pipeline2(preprocessor):

    xgb_pipeline2 = Pipeline([('preprocessor', preprocessor), ('select', SelectPercentile(percentile=33)),
                             ('model', XGBClassifier(random_state=2))])
    return xgb_pipeline2


def load_kmeans():
    kmeans = pickle.load(open("models/kmeans.sav", 'rb'))
    return kmeans


def load_X_test():
    X_test = pd.read_csv("data/X_test.csv")
    return X_test

# @st.cache
def load_X_train():
    X_train = pd.read_csv("data/X_train.csv")
    return X_train


def load_y_test_benign(df_labels):
    y_test_benign = list(df_labels['y_test_benign'])
    return y_test_benign


def load_y_test_preds_new(df_labels):
    y_test_preds_new = list(df_labels['y_test_preds_new'])
    return y_test_preds_new


def load_kmeans_test_y(df_labels):
    kmeans_test_y = list(df_labels['kmeans_test_y'])
    return kmeans_test_y


def load_y_test_preds_combined(df_labels):
    y_test_preds_combined = list(df_labels['y_test_preds_combined'])
    return y_test_preds_combined

def load_test_probabilities(df_labels):
    test_probabilities = list(df_labels['test_probabilities'])
    return test_probabilities


def load_X_train_new2():
    X_train_new2 = read_array('data/X_train_new2.txt')
    return X_train_new2


def load_X_test_new2():
    X_test_new2 = read_array('data/X_test_new2.txt')
    return X_test_new2


def load_X_train_new():
    X_train_new = read_array('data/X_train_new.txt')
    return X_train_new


def load_X_test_new():
    X_test_new = read_array('data/X_test_new.txt')
    return X_test_new


def load_continuous_cols():
    continuous_cols = list(pd.read_csv('data/continuous_cols.csv')['continuous_cols'])
    return continuous_cols


def get_present_column_subset(selected_columns, df):
    # get the intersecton of present and known-infrequent columns
    present_columns = df.columns
    cols = [col for col in present_columns if col in selected_columns]
    return cols

def load_kmeans_test_labels():
    df = pd.read_csv('data/kmeans_test_labels.csv')
    kmeans_test_labels = list(df['kmeans_test_labels'])
    return kmeans_test_labels

def load_df_crosstab():
    df_crosstab = pd.read_csv('data/df_crosstab.csv')
    return df_crosstab

@st.cache
def load_X_test_malicious(X_test_answer, test_probabilities, y_test_preds_combined):
    scaler = MinMaxScaler((1, 10))
    X_test_answer['risk_score'] = test_probabilities
    mal_dict = {0.0: 'benign', 1.0: 'malicious'}
    X_test_answer['y_test_preds_combined'] = [mal_dict[x] for x in y_test_preds_combined]
    X_test_answer['risk_score'] = scaler.fit_transform(X_test_answer[['risk_score']])
    X_test_malicious = X_test_answer[X_test_answer['y_test_preds_combined'] == 'malicious']
    X_test_malicious['Severity'] = X_test_malicious['risk_score'].apply(lambda x: get_severity(x))
    X_test_malicious = X_test_malicious.sort_values(by='sample_id', ascending=True).reset_index()
    return X_test_malicious
@st.cache
def load_data():
    continuous_cols = load_continuous_cols()

    df_labels = pd.read_csv('data/labels.csv')

    kmeans = load_kmeans()
    X_test = load_X_test()

    kmeans_test_y = load_kmeans_test_y(df_labels)
    y_test_preds_new = load_y_test_preds_new(df_labels)
    y_test_benign = load_y_test_benign(df_labels)
    y_test_preds_combined = load_y_test_preds_combined(df_labels)
    test_probabilities = load_test_probabilities(df_labels)
    X_train_new2 = load_X_train_new2()
    X_test_new2 = load_X_test_new2()
    X_train_new = load_X_train_new()
    X_test_new = load_X_test_new()
    X_train = load_X_train()
    df_test_labels = pd.read_csv('data/df_test_labels.csv')
    y_train_benign = list(df_test_labels['y_train_benign'])
    y_train = df_test_labels['y_train']

    X_test.insert(0, 'sample_id', ['SID' + str(x + 1) for x in X_test.index])
    X_test_answer = X_test.copy()

    X_test_malicious = load_X_test_malicious(X_test_answer, test_probabilities, y_test_preds_combined)

    kmeans_test_labels = load_kmeans_test_labels()
    df_crosstab = load_df_crosstab()


    return continuous_cols, kmeans, X_test, X_test_malicious, kmeans_test_y, y_test_preds_new, y_test_benign, \
y_test_preds_combined, X_train_new2, X_test_new2, X_train_new, X_test_new, X_train, y_train_benign, y_train, kmeans_test_labels, df_crosstab

def get_severity(risk):
    if 0 <= risk <= 3.9:
        return 'Low'
    elif 4.0 <= risk <= 6.9:
        return 'Medium'
    elif 7.0 <= risk <= 8.9:
        return 'High'
    elif 9.0 <= risk <= 10.0:
        return 'Critical'


def explain_instance(i, kmeans, kmeans_test_y, y_test_preds_new, kmeans_test_labels, y_test_benign,
                     y_test_preds_combined, df_crosstab, continuous_cols, X_test, X_train, xgb_pipeline,
                     xgb_pipeline2, X_train_new, X_test_new, X_test_new2, col1, col2,
                     optimal_threshold = 0.0050058886):

    remediation_dict = {'dos': 'Для атак типа «Отказ в обслуживании» (DoS) мы рекомендуем проанализировать '
                               'журналы сетевого трафика, чтобы подтвердить эту атаку. Если подтвержден DoS,'
                               ' немедленно свяжитесь со службой реагирования на инциденты, чтобы держать под контролем'
                               ' и блокировать нежелательный сетевой трафик',
                        'r2l': 'Для атак типа Remote to Local (R2L) мы рекомендуем определить адрес удаленной машины и подтвердить факт нелегетимного подключения, '
                               'временно отключить возможность удаленного подключения к сети, а затем найти и устранить все бэкдоры и точки доступа.',
                        'u2r': 'Для атак типа User to Root (U2R) мы рекомендуем определить источник этого трафика, изолировать хост (если возможно) '
                               'заблокировать трафик и провести тщательную проверку безопасности на наличте вредоносных программ, взломов учетных записей и т.д.',
                        'probe': 'Для атак типа Probe мы рекомендуем для начала проверить сетевой трафик на наличие вторжения и после чего '
                                 'настроить межсетевой экран для предотвращения сканирования ваших серверов.',
                        'Outlier': 'Для неопознанных атак мы рекомендуем дополнительно проанализировать выборку в таблице выше, которая сможет помочь '
                                   'отделу безопасности правильно классифицировать атаку. Также продолжайте отслеживать сетевой трафик на наличие аномалий.'}



    class_dict = {'benign': 'Benign', 'dos': 'DoS', 'r2l': 'R2L', 'u2r': 'U2R', 'probe': 'Probe', 'Outlier': 'Outlier'}
    mal_dict = {0.0: 'Benign', 1.0: 'Malicious'}

    if kmeans_test_y[i] > y_test_preds_new[i]:

        df_cluster_weights = pd.DataFrame(columns=continuous_cols, data=kmeans.cluster_centers_)
        cg = sns.clustermap(df_cluster_weights, vmin=-3, vmax=3)
        row_order = cg.dendrogram_row.reordered_ind
        plot = plt.setp(cg.ax_heatmap.yaxis.get_majorticklabels(), rotation=0)
        cg.ax_heatmap.add_patch(
            Rectangle((0, row_order.index(kmeans_test_labels[i])), 31, 1, fill=False, edgecolor='yellow', lw=4,
                      clip_on=False))
        hm = cg.ax_heatmap.get_position()
        cg.ax_heatmap.tick_params(length=0)
        cg.ax_heatmap.set_position([hm.x0, hm.y0 + hm.height - (hm.height*.75), hm.width, hm.height*.75])
        st.pyplot(cg)

        if df_crosstab.loc[kmeans_test_labels[i]]['outlier'] == True:
            reason = 'Outlier'
        else:
            reason = df_crosstab.loc[kmeans_test_labels[i]]['max']

        feature_tup = sorted(dict(df_cluster_weights.iloc[kmeans_test_labels[i]]).items(), key=lambda x: abs(x[1]),
                             reverse=True)
        names = [x[0] for x in feature_tup][:10]
        vals = [X_test.iloc[i][x[0]] for x in feature_tup][:10]
        df_features = pd.DataFrame({'Feature': names, 'Original Values': vals})
        with col2:
            st.dataframe(df_features, height = 200)
        with col1:
            st.markdown(
                '<p style="font-family:sans-serif; color:#707070; font-size: 20px;">Категория</p>',
                unsafe_allow_html=True)
            st.markdown('<p style="font-family:sans-serif; color:#A00000; font-size: 30px;"><i>%s</i></p>' % (
                class_dict[reason]), unsafe_allow_html=True)
        st.write('Приведенный пример определяется как',
              str(mal_dict[y_test_preds_combined[i]]) + '. Это частично объясняется значениями', feature_tup[0][0],
              'of', str(X_test.iloc[i][feature_tup[0][0]]), ',', feature_tup[1][0], '=', str(X_test.iloc[i][feature_tup[1][0]])+
              ', и', feature_tup[2][0], '=', str(X_test.iloc[i][feature_tup[2][0]]) + '.' + ' Данная атака, скорее всего, классифицируется как', reason + '.')
        st.markdown('#### Рекомендации')
        st.write(remediation_dict[reason])
    else:
        explainer = lime.lime_tabular.LimeTabularExplainer(X_train_new, class_names=['Benign', 'Malicious'], feature_names=list(
            X_train.columns[xgb_pipeline.steps[1][1].get_support()]), discretize_continuous=False,
                                                           mode='classification')
        exp = explainer.explain_instance(X_test_new[i], xgb_pipeline.steps[2][1].predict_proba, num_features=10)
        feature_tup = exp.as_list()
        plot = plt.figure(figsize=(4,3))
        vals = [x[1] for x in feature_tup]
        names = [x[0] for x in feature_tup]
        vals.reverse()
        names.reverse()
        colors = ['red' if x > 0 else 'green' for x in vals]
        pos = np.arange(len(feature_tup)) + .5
        plt.barh(pos, vals, align='center', color=colors)
        plt.yticks(pos, names)
        title = 'Local explanation for class %s' % mal_dict[y_test_preds_combined[i]]
        plt.title(title)

        names = [x[0] for x in feature_tup]
        vals = [X_test.iloc[i][x[0]] for x in feature_tup]
        df_features = pd.DataFrame({'Feature': names, 'Original Values': vals})
        with col2:
            st.dataframe(df_features, height = 200)
        st.pyplot(plot)


        if y_test_preds_combined[i] == 1.0:
            if xgb_pipeline2.steps[2][1].predict(X_test_new2[[i]])[0] != 'benign':
                category = xgb_pipeline2.steps[2][1].predict(X_test_new2[[i]])[0]
            else:
                probs = dict(zip(xgb_pipeline2.steps[2][1].classes_, list(xgb_pipeline2.steps[2][1].predict_proba(X_test_new2[[i]])[0])))
                probs = sorted(probs.items(), key = lambda x: x[1], reverse = True)
                category = probs[1][0]
            with col1:
                st.markdown('<p style="font-family:sans-serif; color:#707070; font-size: 20px;">Категория</p>',
                    unsafe_allow_html=True)
                st.markdown('<p style="font-family:sans-serif; color:#A00000; font-size: 30px;"><i>%s</i></p>' % (
                    class_dict[category]), unsafe_allow_html=True)

            st.write('Приведенный пример определяется как',
                     str(mal_dict[y_test_preds_combined[i]]) + '. Это частично объясняется значениями',
                     feature_tup[0][0], '=', str(X_test.iloc[i][feature_tup[0][0]]), ',', feature_tup[1][0], '=',
                     str(X_test.iloc[i][feature_tup[1][0]]) + ', и', feature_tup[2][0], '=',
                     str(X_test.iloc[i][feature_tup[2][0]]) + '.' + ' Данная атака, скорее всего, классифицируется как',
                     class_dict[category] + '.')

            st.markdown('#### Рекомендации')
            st.write(remediation_dict[category])


