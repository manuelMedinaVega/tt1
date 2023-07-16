options(repos = c(CRAN = "http://cran.rstudio.com"))

install.packages("tidyverse", dependencies = TRUE)
install.packages("tidymodels", dependencies = TRUE)
install.packages("rsample", dependencies = TRUE)
install.packages("corrr", dependencies = TRUE)
install.packages("knitr", dependencies = TRUE)
install.packages("kableExtra", dependencies = TRUE)
install.packages("GGally", dependencies = TRUE)
install.packages("robustbase", dependencies = TRUE)
install.packages("reshape", dependencies = TRUE)
install.packages("dplyr", dependencies = TRUE)
install.packages("ggcorrplot", dependencies = TRUE)

# Carga de liberías
library(tidyverse)
library(tidymodels)
library(rsample)
library(corrr)
library(knitr)
library(kableExtra)
library(GGally)
library(robustbase)
library(reshape)
library(dplyr)
library(ggcorrplot)
library(plyr)
library(caret)

set.seed(491)

#LECTURA DE LOS DATOS
#data <- read.csv("C:/Users/programadorweb4/Documents/m_d_m/tt1/tt1/train.csv", 
  #row.names = 'id');
data <- read.csv("C:/Users/programadorweb4/Documents/m_d_m/tt1/tt1/train.csv");
head(data)
testdata <- read.csv("C:/Users/programadorweb4/Documents/m_d_m/tt1/tt1/test.csv");
head(testdata)
countTrain <- count(data)
countTest <- count(testdata)
percTest <- countTest * 100 / (countTrain + countTest)
percTest
0.3*countTrain
#el dataset de test no cuenta con la variable respuesta, por lo que dividimos
#el dataset de train en train-test
#convertimos a factor los datos categóricos
data <- data %>%
  mutate(Gender = as.factor(Gender), 
         Driving_License = as.factor(Driving_License),
         Region_Code = as.factor(Region_Code), 
         Previously_Insured = as.factor(Previously_Insured), 
         Vehicle_Age = as.factor(Vehicle_Age),
         Vehicle_Damage = as.factor(Vehicle_Damage),
         Policy_Sales_Channel = as.factor(Policy_Sales_Channel),
         Response = as.factor(Response))
#resumen de los datos
data %>% glimpse()
#-------------------------------------------------------------------------------

#VALORES ÚNICOS Y FALTANTES
tabla_exploratorios =  data %>%
  gather(., 
    key = "variables", 
    value = "valores"
  ) %>% # agrupamos por las variables del set
  group_by(variables) %>% 
  summarise(
    "valores únicos" = n_distinct(valores),
    porcentaje_faltantes = sum(is.na(valores))/nrow(data)*100
  ) %>% 
  arrange(
    desc(porcentaje_faltantes), 
    "valores únicos"
  ) # ordenamos por porcentaje de faltantes y valores unicos
tabla_exploratorios
#otra forma de ver valores faltantes
sapply(data, function(x) sum(is.na(x)))
#cantidad de registros en cada categoría de la variable Policy_Sales_Channel
#count(data, "Policy_Sales_Channel") #usa la función de plyr y 
#no puede usarse junto con dply, sino no funciona tabla_exploratorias
#------------------------------------------------------------------------------


#RESUMEN ESTADÍSTICO DE LOS DATOS
#muestra de los datos
set.seed(491)
sdata <- data %>% sample_n(38000)
summary(select(sdata, -id))
#datos numéricos
sdata %>% select(Age, Annual_Premium, Vintage) %>% summary
#datos categóricos
sdata %>% select(-id, -Age, -Annual_Premium, -Vintage) %>% summary
sdata %>% 
  select(Gender, Driving_License, Previously_Insured, Vehicle_Damage) %>% summary
sdata %>% 
  select(Region_Code, Policy_Sales_Channel, Vehicle_Age, Response) %>% summary
#------------------------------------------------------------------------------

#GRÁFICOS DE VARIABLES NUMÉRICAS
#histograma age
ggplot(sdata, aes(x = Age)) + 
  geom_histogram(binwidth = 1) +
  labs(y="Cantidad")
#gráfico de densidad Age
ggplot(sdata, aes(x = Age, color=Response)) + 
  geom_density() +
  labs(y="Densidad")
#gráfico de densidad Annual_Premium
#de esta mejor mostrar boxplot
ggplot(sdata, aes(x = Annual_Premium, color = Response)) + 
  geom_boxplot()
#gráfico de densidad Vintage
#mejor no mostrar la de vintage
ggplot(sdata, aes(x = Vintage, color=Response)) + 
  geom_density() +
  labs(y="Densidad")
#correlación, heatmap
sdata_num <- sdata %>% mutate(Response = as.numeric(Response))
corr <- cor(sdata_num %>% select(Age, Annual_Premium, Vintage, Response))
ggcorrplot(corr, lab = TRUE)
#ggpairs
ggpairs(sdata %>% select(Age, Annual_Premium, Vintage, Response), 
        mapping = ggplot2::aes(color = Response, alpha = 0.1), 
        lower = list(combo = "box"),progress = F) +
  theme_bw() + 
  theme(axis.text.x = element_text(angle=50, vjust=0.5), 
        legend.position = "bottom") 
labs(title= 'Variables numéricas', x='Variable', y='Variable')
#------------------------------------------------------------------------------

#VARIABLES CATEGÓRICAS
sdata_cat <- sdata %>% select(Response, 
                              Gender, 
                              Driving_License, 
                              Previously_Insured, 
                              Vehicle_Age, 
                              Vehicle_Damage)
#sdata_cat <- melt(sdata_cat)
#gráfico barras gender-response
ggplot(sdata_cat) +
  geom_bar(aes(x = Gender, fill = Response), color = 'lightblue') +
  labs(x = 'Gender', y = 'Cantidad') +
  theme(axis.text.x = element_text(face = 'bold', size = 10),
        axis.text.y = element_text(face = 'bold', size = 10))
#gráfico de barras Driving_License-response
ggplot(sdata_cat) +
  geom_bar(aes(x = Driving_License, fill = Response), color = 'lightblue') +
  labs(x = 'Driving_License', y = 'Cantidad') +
  theme(axis.text.x = element_text(face = 'bold', size = 10),
        axis.text.y = element_text(face = 'bold', size = 10))
#gráfico de barras Previously_Insured-response
ggplot(sdata_cat) +
  geom_bar(aes(x = Previously_Insured, fill = Response), color = 'lightblue') +
  labs(x = 'Previously_Insured', y = 'Cantidad') +
  theme(axis.text.x = element_text(face = 'bold', size = 10),
        axis.text.y = element_text(face = 'bold', size = 10))
#gráfico de barras Vehicle_Age-response
ggplot(sdata_cat) +
  geom_bar(aes(x = Vehicle_Age, fill = Response), color = 'lightblue') +
  labs(x = 'Vehicle_Age', y = 'Cantidad') +
  theme(axis.text.x = element_text(face = 'bold', size = 10),
        axis.text.y = element_text(face = 'bold', size = 10))
#gráfico de barras Vehicle_Damage-response
ggplot(sdata_cat) +
  geom_bar(aes(x = Vehicle_Damage, fill = Response), color = 'lightblue') +
  labs(x = 'Vehicle_Damage', y = 'Cantidad') +
  theme(axis.text.x = element_text(face = 'bold', size = 10),
        axis.text.y = element_text(face = 'bold', size = 10))
#------------------------------------------------------------------------------

#FILTROS PARA CONTAR EN CATEGORÍAS
#response
nR1 <- data %>% filter(Response == '1') %>% nrow
nR1
nR0 <- data %>% filter(Response == '0') %>% nrow
nR0
count(data)
nR1*100/count(data)
#gender
nMale <- sdata_cat %>% filter(Gender == 'Male') %>% nrow
nMale
nFemale <- sdata_cat %>% filter(Gender == 'Female') %>% nrow
nFemale
sdata_cat %>% filter(Gender == 'Male') %>% 
  filter(Response == '1') %>% nrow * 100 / nMale
sdata_cat %>% filter(Gender == 'Female') %>% 
  filter(Response == '1') %>% nrow * 100 / nFemale

#driving_license
nNoDrivLic <- sdata_cat %>% filter(Driving_License == '0') %>% nrow
nNoDrivLic
nSisDrivLic <- sdata_cat %>% filter(Driving_License == '1') %>% nrow
nSisDrivLic
sdata_cat %>% filter(Driving_License == '0') %>% 
  filter(Response == '1') %>% nrow * 100 / nNoDrivLic
sdata_cat %>% filter(Driving_License == '1') %>% 
  filter(Response == '1') %>% nrow * 100 / nSisDrivLic

#vehicle_age
nMenor1 <- sdata_cat %>% filter(Vehicle_Age == '< 1 Year') %>% nrow
nMenor1
nMayor2 <- sdata_cat %>% filter(Vehicle_Age == '> 2 Years') %>% nrow
nMayor2
n12Year <- sdata_cat %>% filter(Vehicle_Age == '1-2 Year') %>% nrow
n12Year
sdata_cat %>% filter(Vehicle_Age == '1-2 Year') %>%
  filter(Response == 1) %>% nrow

#previously_insured
nPrevIns <- sdata_cat %>% filter(Previously_Insured == 1) %>% nrow
nPrevIns
nNoPrevIns <- sdata_cat %>% filter(Previously_Insured == 0) %>% nrow
nNoPrevIns
sdata_cat %>% filter(Previously_Insured == 1) %>% 
  filter(Response == 1) %>% nrow
sdata_cat %>% filter(Previously_Insured == 0) %>% 
  filter(Response == 1) %>% nrow * 100 / nNoPrevIns

#vehicle_damage
nVehDam <- sdata_cat %>% filter(Vehicle_Damage == 'Yes') %>% nrow
nVehDam
nNoVehDam <- sdata_cat %>% filter(Vehicle_Damage == 'No') %>% nrow
nNoVehDam
sdata_cat %>% filter(Vehicle_Damage == 'Yes') %>% 
  filter(Response == 1) %>% nrow * 100 / nVehDam
sdata_cat %>% filter(Vehicle_Damage == 'No') %>% 
  filter(Response == 1) %>% nrow * 100 / nVehDam
#------------------------------------------------------------------------------


#PRE PROCESAMIENTO DE LOS DATOS
#one-hot-encoding para xgboost
oh_sdata <- sdata
oh_sdata <- oh_sdata %>% mutate(Response = as.numeric(Response))
dummies <- dummyVars("~.", data = oh_sdata, fullRank = T)
train_dummies <- data.frame(predict(dummies, newdata = oh_sdata))
train_dummies


