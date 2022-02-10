from django.shortcuts import render, HttpResponse
from .randomforest import predict_out_Random
from .logistic import predict_out_logistic
from .svm import predict_out_SVM
from .GaussianNB import predict_out_GNB

# Create your views here.
def index(request):
    # return HttpResponse("this is homepage")
    return render(request, 'homepage.html')

def form(request):
    return render(request,'form.html')
def predict(request):
    if request.method == "POST":
        test={}

        test['albumin'] = request.POST.get('albumin')
        test['alkaline_phosphatase'] = request.POST.get('alkaline_phosphatase')
        test['alanine_aminotransferase'] = request.POST.get('alanine_aminotransferase')
        test['aspartate_aminotransferase'] = request.POST.get('aspartate_aminotransferase')
        test['bilirubin'] = request.POST.get('bilirubin')

        test['cholinesterase'] = request.POST.get('cholinesterase')
        test['cholesterol'] = request.POST.get('cholesterol')

        test['creatinine'] = request.POST.get('creatinine')

        test['gamma_glutamyl_transferase'] = request.POST.get('gamma_glutamyl_transferase')
        test['protein'] = request.POST.get('protein')
        model = request.POST.get('model')

        print(test)
        if model=="Random Forest":
            out = predict_out_Random(test)

        elif model=="Logistic Regression":
            out = predict_out_logistic(test)

        elif model=="SVM":
            out = predict_out_SVM(test)

        else:
            out = predict_out_GNB(test)

        print(out)

        data = {'output': out}


    return render(request,'form.html',data)