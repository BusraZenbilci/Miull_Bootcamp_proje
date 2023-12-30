import streamlit as st
import pandas as pd
import joblib
from sklearn.preprocessing import LabelEncoder


df = pd.read_csv('dataset/loan_approval_dataset.csv')

def load_model():
    model = joblib.load("rf_model.joblib")
    return model

def get_user_input():
    CIBIL_SCORE = st.slider("Kredi Puanı", 300, 900, 600, step=1)
    LOAN_TERM = st.slider("Kredi Süresi (Yıl)", 2, 20, 10, step=1)
    INCOME_ANNUM = st.number_input("Yıllık Gelir", min_value=10000, value=50000)
    NO_OF_DEPENDENTS = st.number_input("Bakmakla Yükümlü Olduğunuz Kişi Sayısı", 0, 30, 0, step=1)
    SELF_EMPLOYED = st.radio("Çalışma Durumu", ["Yes", "No"])
    EDUCATION = st.radio("Eğitim Durumu", ["Graduate", "Not Graduate"])
    LOAN_AMOUNT = st.number_input("İstenilen Kredi Miktarı", min_value=100000, value=500000)

    user_input = pd.DataFrame({
        'CIBIL_SCORE': [CIBIL_SCORE],
        'LOAN_TERM': [LOAN_TERM],
        'INCOME_ANNUM': [INCOME_ANNUM],
        'NO_OF_DEPENDENTS': [NO_OF_DEPENDENTS],
        'SELF_EMPLOYED': [SELF_EMPLOYED],
        'EDUCATION': [EDUCATION],
        'LOAN_AMOUNT': [LOAN_AMOUNT]
    })

    return user_input


def label_encode(dataframe, column, le):
    dataframe[column] = le.transform(dataframe[column].astype(str))
    return dataframe


def main():
    st.set_page_config(layout="wide")
    st.title("💵💶💷 MIUULBANK KREDİ ONAY TAHMİNİ 💵💶💷")

    tabs = ["Ana Sayfa", "Grafikler", "Model"]
    selected_tab = st.sidebar.radio("Sayfa Seç", tabs)

    # TAB HOME
    if selected_tab == "Ana Sayfa":
        column1, column2 = st.columns([3, 2])
        column1.subheader("1. Proje konusu")
        column1.markdown("Bu projede hedeflenen amaç makine öğrenimi kullanılarak müşterilerden elde edilen bir takım kişisel "
                         "bilgilerle bankanın müşterinin istediği kredi miktarını onaylanıp onaylanmayacağını tahmin etmek.")
        column2.image("media/bank.png", width=400)
        column1.subheader("2.Veri seti")
        column1.markdown("Veri seti 4269 gözlem ve 12 bağımsız değişkenden oluşmaktadır. Hedef değişken 'loan_status' olarak "
            "belirtilmiş olup 'Approved' (0) kredinin onaylandığını, 'Rejected' 1 ise kredinin onaylanmadığını "
            "belirtmektedir. Değişkenler ve açıklamaları aşağıdaki gibidir: ")

        column1.markdown("""
        * loan_id : Her bir müşterinin benzersiz kimlik numarası. 
        * no_of_dependents : Başvuru sahibinin bakmakla yükümlü olduğu kişi sayısı.
        * education : Başvuru sahibinin eğitim seviyesi, Lisansüstü ya da Lisansüstü değil.
        * self_employed : Başvuru sahibinin serbest meslek sahibi olup olmadığı.
        * income_annum : Başvuru sahibinin yıllık geliri.
        * loan_amount : Kredi için talep edilen toplam tutar.
        * loan_term : Kredinin geri ödenmesi gereken yıl cinsinden süre.
        * cibil_score : Başvuru sahibinin kredi puanı.
        * residential_assets_value : Başvuru sahibinin konut varlıklarının toplam değeri.
        * commercial_assets_value : Başvuru sahibinin ticari varlıklarının toplam değeri.
        * luxury_assets_value : Başvuru sahibinin lüks varlıklarının toplam değeri.
        * bank_asset_value : Başvuru sahibinin banka varlıklarının toplam değeri.
        * loan_status : Hedef değişken. Kredinin onaylanıp onaylanmadığını açıklar.
        """)

        st.dataframe(df)

    # TAB VIS
    elif selected_tab == "Grafikler":

        st.subheader("Veri seti üzerine ilgili grafikler:")

        column1, column2 = st.columns([1,1])

        column1.markdown("Kredi verilen ve verilmeyen müşterilerin sayısı: ")
        column1.image("media/image 3.png", width = 400)

        column2.markdown("Özniteliklerin kredi üzerindeki etkileri: ")
        column2.image("media/feature_importance.png", width = 400)

        column1.subheader("Farklı modellerin onfusion matrix")

        column1.markdown("XGBOOST confusion matrix")
        column1.image("media/XGboost_confusion_matrix.jpg")
        column1.markdown("""
        * Accuracy : 0.956
        * Recall : 0.982
        * Precision : 0.945
        * F1_Score : 0.963
        """)

        column2.markdown("LIGHTGBM confusion matrix")
        column2.image("media/lightGBM_confusion_matrix.jpg")
        column2.markdown("""
        * Accuracy : 0.971
        * Recall : 0.968
        * Precision : 0.985
        * F1_Score : 0.976
        """)

        column1.markdown("CATBOOST confusion matrix")
        column1.image("media/catboost_confusion_matrix.jpg")
        column1.markdown("""
        * Accuracy : 0.966
        * Recall : 0.964
        * Precision : 0.981
        * F1_Score : 0.972
        """)

        column2.markdown("RANDOM FOREST confusion matrix")
        column2.image("media/randomForest_confusion_matrix.jpg")
        column2.markdown("""
        * Accuracy : 0.971
        * Recall : 0.972
        * Precision : 0.981
        * F1_Score : 0.976
        """)

        column1.markdown("Random forest performance ")
        column1.image("media/randomForest_performance.jpg")

        column1.markdown("Correlation matrix")
        column1.image("media/correlation_matrix.png", width = 700)

        pass

    # TAB MODEL
    elif selected_tab == "Model":
        st.header("Lütfen aşağıya bilgilerini giriniz: ")

        user_input = get_user_input()

        le_self_employed = LabelEncoder()
        le_education = LabelEncoder()

        # Orijinal veri setindeki sütunları kullan
        df["SELF_EMPLOYED"] = le_self_employed.fit_transform(df["SELF_EMPLOYED"])
        df["EDUCATION"] = le_education.fit_transform(df["EDUCATION"])

        # Giriş verisini LabelEncoder'lar kullanarak dönüştür
        user_input = label_encode(user_input, "SELF_EMPLOYED", le_self_employed)
        user_input = label_encode(user_input, "EDUCATION", le_education)

        # Dönüştürülen değerlere göre kategorik değerleri atayın
        user_input["SELF_EMPLOYED"] = "Yes" if user_input["SELF_EMPLOYED"].iloc[0] == 0 else "No"
        user_input["EDUCATION"] = "Graduate" if user_input["EDUCATION"].iloc[0] == 0 else "Not Graduate"

        model = load_model()

        prediction = model.predict(user_input)

        st.header("Tahmin Sonucu")
        if prediction[0] == 0:
            st.success("Kredi Onaylandı!")
        else:
            st.error("Kredi Reddedildi.")

if __name__ == "__main__":
    main()
