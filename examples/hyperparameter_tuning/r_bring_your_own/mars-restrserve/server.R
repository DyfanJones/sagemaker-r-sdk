# server.R
library(RestRserve)
library(data.table)

model_path = file.path(prefix, "model")
load(file.path(model_path, 'model.RData'))

# Extend content_type by adding in custom encoder and decoder
# it is important to enable your RestRserve to accept content_type: "text/csv" as this is
# the default method AWS Sagemaker in CSVSerializer class
encode_decode_middleware = EncodeDecodeMiddleware$new()
encode_decode_middleware$ContentHandlers$set_encode(
  "text/csv",
  encode_decode_middleware$ContentHandlers$get_encode('text/plain')
)

encode_decode_middleware$ContentHandlers$set_decode(
  "text/csv",
  encode_decode_middleware$ContentHandlers$get_decode('text/plain')
)

app = Application$new(middleware = list())
app$add_get("/ping",
            FUN = function(req, res) {
              res$set_body("R6sagemaker mars restrserve example")})

app$add_post("/invocations",
             FUN= function(req, res){

               # added in a optional switch to enable support for content_type:
               #    *  "text/plain"
               #    *  "text/csv"
               #    *  "application/json"
               data = switch(req$content_type,
                             "application/json" = as.data.table(req$body),
                             fread(req$body)) # method to read in "text/csv" data

               result = predict(model, data, row.names=FALSE)
               res$set_body(result)
             }
)

app$append_middleware(encode_decode_middleware)
