from flask import Flask, request, render_template
import processing

app = Flask(__name__,template_folder='template')

@app.route('/')
def my_form():
    return render_template('index.html')

@app.route('/', methods=['POST'])
def my_form_post():
    text= request.form['chapter']
    new_var,K,L,D,R,R_C,S,your_list,graph = processing.pree(text)
    graphJSON = processing.country_graph(K)
    L_graph = processing.generate_graph_language(L)
    D_graph = processing.get_date_time(D)
    R_graph = processing.rating_graph(R)
    sentiment_graph = processing.sentiment_graph(R_C)
    sellers_graph = processing.sellers_graph(S)

    return render_template('index.html',new_var = text,graphJSON =graphJSON, num_of_review = new_var,
                           L_graph = L_graph, D_graph = D_graph, R_graph = R_graph, RC_graph = sentiment_graph,sellers_graph = sellers_graph,your_list = your_list, geo_graph = graph)






app.run()