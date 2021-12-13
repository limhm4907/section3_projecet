from flask import Flask, render_template, request
import pickle


def create_app():
    
    app = Flask(__name__)

    # 모델 가져오기
    model = None
    with open('model.pkl','rb') as pickle_file:
        model = pickle.load(pickle_file)

    @app.route('/')
    def indef():
        return render_template('index.html')

    
    @app.route('/result', methods=['POST']) #, methods=['POST','GET']
    def result():
        
        # q1, q2
        q1_is_checked = request.form.getlist('q1')
        q2_is_checked = request.form.get('q2')

        if 'phone' in q1_is_checked:
            q1_phone = 2
        else:
            q1_phone = 1

        if 'internet' in q1_is_checked:
            q2_internet = int(q2_is_checked)
        else:
            q2_internet = 1

        # q3
        q30_is_checked = request.form.getlist('q30')
        q31_is_checked = request.form.getlist('q31')

        if q30_is_checked[0] == 'q30_disabled':
            q3_numadd = 0
        else:
            q3_numadd = len(q31_is_checked)
        
        # q4, q5, q6, q7
        q4_charge = request.form.get('q4')
        q5_payment = request.form.get('q5')
        q6_contract = request.form.get('q6')
        q7_tenure = request.form.get('q7')

        # text -> float/int
        q4_flo = float(q4_charge)
        q5_int = int(q5_payment)
        q6_int = int(q6_contract)
        q7_int = int(q7_tenure)

        X_input = [[q7_int, q1_phone, q2_internet, q6_int, q5_int, q4_flo, q3_numadd]]
        y_pred = model.predict(X_input)

        # 예측 결과 출력하기
        return render_template('index.html', final=y_pred)

        # 입력된 데이터 출력하기 test -> check done
        # return render_template('index.html', ans1=q1_phone, ans2=q2_internet, ans3=q3_numadd, ans4=q4_charge, ans5=q5_payment, ans6=q6_contract, ans7=q7_tenure)
        # return render_template('index.html', ans1=type(q1_phone), ans2=type(q2_internet), ans3=type(q3_numadd), ans4=type(q4_charge), ans5=type(q5_payment), ans6=type(q6_contract), ans7=type(q7_tenure))
        # return render_template('index.html', ans1=q1_phone, ans2=q2_internet, ans3=q3_numadd, ans4=q4_flo, ans5=q5_int, ans6=q6_int, ans7=q7_int)
        
    return app

# if __name__ == '__main__':
#     app.run(debug=True)