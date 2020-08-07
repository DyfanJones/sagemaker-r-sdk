# This example uses rstudio r-base docker images: https://github.com/rstudio/r-docker
FROM rstudio/r-base:4.0.2-centos7

COPY . /tmp/RestRserve

RUN yum -y update && \
    yum -y install openssl-devel

# install dependency packages
RUN R -e "install.packages(c('Rcpp','R6','uuid','checkmate','mime','jsonlite','data.table','mda'), repos='https://cloud.r-project.org')"

# install Rserve and RestRserve
RUN R -e "install.packages('Rserve', repos = 'http://www.rforge.net/')" && \
    R -e "install.packages('RestRserve', repos='https://cloud.r-project.org')"

COPY mars.R /opt/ml/mars.R
COPY server.R /opt/ml/server.R

EXPOSE 8080

ENTRYPOINT ["/usr/bin/Rscript", "/opt/ml/mars.R", "--no-save"]
