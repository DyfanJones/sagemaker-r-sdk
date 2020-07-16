# server.R
library(RestRserve)
library(data.table)

# prefix = '/opt/ml'
model_path = file.path(prefix, "model")
load(file.path(model_path, 'model.RData'))

app = Application$new()
app$add_get("/ping",
            FUN = function(req, res) {
              res$set_body("")})

app$add_post("/model",
             FUN= function(req, res){

               data = fread(req$body)

               # # Convert input to model matrix
               scoring_X <- model.matrix(~., data, xlev=factor_levels)

               # res$content_type = "application/json"
               result = predict(model, scoring_X, row.names=FALSE)
               res$set_body(result)
             }
)
