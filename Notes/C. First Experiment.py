import mlflow
import pandas as pd

# with mlflow.start_run(run_name = "Logging Demo"):
#     mlflow.log_param("learning_rate",0.03)
#     mlflow.log_param('epoch',100)

#     params_dict = {
#         'learning_rate1':0.03,
#         'epoch1':100,
#         'optimizer':'rad'
#     }
#     mlflow.log_params(params_dict)


with mlflow.start_run(run_name="Logging Demo", run_id="eb0c2fca189340e396c6063896a464cc") as run:
    mlflow.log_metric("accuracy",0.3)   
    metric_dict = {
        'fscore':0.03
    }

    mlflow.log_metrics(metric_dict)

    #Artifacts
    artifact_path = './requirements.txt'
    mlflow.log_artifact(artifact_path)

    #Other methods
    demo_df = pd.DataFrame({'name': ['Daniel','Sam']})
    mlflow.log_table(demo_df,'demo_df.json')

    #images
    import matplotlib.pyplot as plt
    plt.plot([1,2,3,4])
    plt.ylabel('some numbers')
    plt.savefig('test.png')
    mlflow.log_figure(plt.gcf(),'test.png')

    #Loaded images
    file_path = "./screenshots/image_1773299179154.png"    
    mlflow.log_image(mlflow.Image(file_path),'notes.png')    

    #Assessments    
    

print("run completed")    