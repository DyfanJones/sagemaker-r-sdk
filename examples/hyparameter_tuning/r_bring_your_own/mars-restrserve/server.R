# server.R
library(RestRserve)
library(data.table)

model_path = file.path(prefix, "model")
load(file.path(model_path, 'model.RData'))

app = Application$new()
app$add_get("/ping",
            FUN = function(req, res) {
              res$set_body("R6sagemaker mars restrserve example")})

app$add_post("/invocations",
             FUN= function(req, res){
               data = fread(req$body)

               result = predict(model, data, row.names=FALSE)
               res$set_body(result)
             }
)
