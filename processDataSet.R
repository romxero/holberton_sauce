#!/usr/bin/env R
#
#
# doing it the holby way 
#
#
#
#
#Testing the Dataset
#install.packages("readxl")

install.packages("tidyverse")
library(readxl)

mentorDataLocation = "/home/romxero/Downloads/lxai_mentor_applications_2021(Fall).xlsx"

menteeDataLocation = "/home/romxero/Downloads/lxai_mentee_applications_2021(Fall).xlsx"

mentorDataSetRaw <- read_excel(path=mentorDataLocation)

menteeDataSetRaw <- read_excel(path=menteeDataLocation)


#mentee data 
menteeDataSetRaw[grep("Female",menteeDataSetRaw$`Gender Self-Identification`),]
menteeDataSetRaw[grep("Male",menteeDataSetRaw$`Gender Self-Identification`),]

#finding locations for mentee data 
locationOfMales = grep("Female",menteeDataSetRaw$`Gender Self-Identification`)
locationOfFemales = grep("Female",mentorDataSetRaw$`Gender Self-Identification`)


#mentor data
mentorDataSetRaw[grep("Male",mentorDataSetRaw$`Gender Self-Identification`),]
mentorDataSetRaw[grep("Female",mentorDataSetRaw$`Gender Self-Identification`),]

#finding locations to generate new data set 
locationOfMales = grep("Male",mentorDataSetRaw$`Gender Self-Identification`)
locationOfFemales = grep("Female",mentorDataSetRaw$`Gender Self-Identification`)


#find the unique values here 
uniqueCountryOfOrigin <- unique(c(menteeDataSetRaw$`Country of Origin`,mentorDataSetRaw$`Country of Origin`))

#this can be used as a guide
countryOfOriginAndCount <- data.frame(uniqueCountryOfOrigin,seq(1,length(uniqueCountryOfOrigin)))

identityMatrice <- unique(mentorDataSetRaw$`Do you identify as being of LatinX origin?`)

#this is to be used to identify if the mentor is an ally, latin, or not
mentorIdentityNumbers <- data.frame(identityMatrice,seq(1,length(identityMatrice)))






