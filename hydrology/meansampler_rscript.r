
# Calculating Mean (Expected) Flow Frequency Curves using Bulletin 17C Confidence Limits
# 1/21/2019

#############
# Step 1. Setup
library(pracma)
options(scipen=999)

#############
# Step 2. Calculate the standard normal z variate for the confidence limits (CL) and the annual exceedence probabilities (AEP) which were used to generate the data file in HEC-SSP; then load in the data file which contains the flows in cfs for each AEP and CL from HEC-SSP, give the column headers the name of the corresponding standard normal x variate for CL, and take the log10 of the flow data:

CL<-c(0.01,0.05,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,0.95,0.99) #Confidence limits
AEP<-c(1E-12, 2E-12, 5E-12, 1E-11, 2E-11, 5E-11, 1E-10, 2E-10, 5E-10 ,1E-09, 2E-09, 5E-09, 1E-08, 2E-08, 5E-08, 1E-07, 2E-07, 5E-07, 1E-06, 2E-06, 5E-06, 1E-05, 2E-05, 5E-05, 1E-04, 2E-04, 5E-04, 0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 0.8, 0.9, 0.95, 0.99) #Annual exceedence probabilities 

CLz=rep(0, length(CL))
for (i in 1:(length(CL))) CLz[i]<-qnorm((1-CL[i])) #Calculate the standard normal z variate

AEPz=rep(0, length(AEP))
for (i in 1:(length(AEP))) AEPz[i]<-qnorm((1-AEP[i])) #Calculate the standard normal z variate

df<-read.csv("sample_data/data.csv", header=FALSE) #Read in the data file
colnames(df)<-CLz
rownames(df)<-AEPz

df1<-df
for (i in 1:(length(CL))) df1[,i]<-log10(df[,i]) #Transform to log10

##############
# Step 3. Plot the data from the table above:

par(mfrow=c(1,2))
pcol <- rep(rainbow(5),length(AEPz))
plot(NA,NA,ylim=c(2.5,6.5),xlim=c(-2.5,7.5), ylab="Log(Q)", xlab="AEP(z)")
#for (i in 1:length(CLz)) abline(v=CLz[i], col="gray")
for(i in 1:length(CLz)){
  lines(AEPz,df1[,i], type="l",col = pcol[i])
  points(AEPz,df1[,i], pch=20,col = pcol[i],cex=0.8)
}
plot(-100,-100,ylim=c(2.5,6.5),xlim=c(-2.5,2.5), ylab="Log(Q)", xlab="CL(z)")
#for (i in 1:length(CLz)) abline(v=CLz[i], col="gray")
for(i in 1:length(AEPz)){
  lines(CLz,df1[i,], type="l",col = pcol[i])
  points(CLz,df1[i,], pch=20,col = pcol[i],cex=0.8)
}

###########
# Step 4. Calculate the minimum and maximum flow values in the data file and then divide this range into equal incruments:

Qmin<-min(df1, na.rm=TRUE)  #The minimum flow at any AEP and CL.
Qmax<-max(df1, na.rm=TRUE) #The maximum flow at any AEP and CL.

bins<-length(AEP) #The number of different AEP's that were calculated which is the number of bins we want

Q<-seq(Qmin,Qmax,length.out=bins) #Linearly interpolate the discharge between the observed max and min discharge with the number of steps equal to the bins number.

###########
# Step 5. Choose if you want to extrapolate the results:
Extrapolate<-"Yes" #'Yes' or 'No'

# Apply linear interpolation/extrapolation to calculate AEP(z) for each CL(z) and binned flow:
df2<-df1

if (Extrapolate == "No") {
    for (j in 1:(length(CLz))){
        #Create a function for linear interpolating the AEPz using logQ and the corresponding AEPz from table 2, only for the first 31 for some reason
        f<-approxfun(df1[1:31,j],AEPz[1:31])

        for (i in 1:(length(Q))){
            # If the binned discharge is less than the minimum discharge for a specific confidence internval than set the estimated AEPz equal to the value for the minimum discharge at that confidence interval
            if (Q[i]<min(df1[1:31,j],na.rm=TRUE)) df2[i,j]<-AEPz[31]

            if ((Q[i]>=min(df1[1:31,j],na.rm=TRUE)) & (Q[i]<=max(df1[1:31,j],na.rm=TRUE))) df2[i,j]<-f(Q[i])
            
            #If the binned discharge is greater than the maximum discharge for a specific confidence internval than set the estimated AEPz equal to the value for the maximum discharge at that confidence interval
            if (Q[i]>max(df1[,j][1:31],na.rm=TRUE)) df2[i,j]<-AEPz[1]

        }
    }
  
# If extrapolation is needed,
#   it cant be done with approxfun()
  
} else { 
    for (j in 1:(length(CLz))){
        # 1. try using approxfun for everything
        f<-approxfun(df1[1:31,j],AEPz[1:31])
        for (i in 1:(length(Q))) df2[i,j]<-f(Q[i])
    }
  
    for (j in 1:(length(CLz))){
        # 2. run through all values and look for NA, which indicates that value needs to be extrapolated
        # 3. if NA is found, replace with a linear model that uses only the endpoints as inputs
        minthresh<-Q[which(df2[,j]==min(df2[,j],na.rm=TRUE))]
        maxthresh<-Q[which(df2[,j]==max(df2[,j],na.rm=TRUE))]
        lowmodel<-lm(y ~ x, data = data.frame(y = c(AEPz[30],AEPz[31]), x = c(df1[30,j],df1[31,j])))
        himodel<-lm(y ~ x, data = data.frame(y = c(AEPz[1],AEPz[2]), x = c(df1[1,j],df1[2,j])))

        for (i in 1:(length(Q))){
            if (is.na(df2[i,j])){
               if (Q[i]<=minthresh) {
                  df2[i,j]<-predict(lowmodel, newdata =  data.frame(x = Q[i]))
               } else {
                  df2[i,j]<-predict(himodel, newdata =  data.frame(x = Q[i]))
               }
            }
        }
    }
}
rownames(df2)<-Q

####################
# Step 6. Plot the data:
x11()
pcol <- rep(rainbow(5),length(Q))

plot(NA,NA,ylim=c(-5,20),xlim=c(-2.5,2.5), ylab="AEP(z)", xlab="CL(z)")
#for (i in 1:length(CLz)) abline(v=CLz[i], col="gray")

for(i in 1:length(Q)) {
  lines(CLz,df2[i,], type="l",col = pcol[i])
  points(CLz,df2[i,], pch=20,col = pcol[i])
}

####################
# Step 7. Take the inverse of the log10 of the discharge and the inverse of the standard normal Z variate for each AEPz. Note the inverse of the standard normal Z variate of CLz is just CL from above:

Q1<-rep(0,length(Q)) #Create an empty array to store the inverse of the log10 of the binned discharge
Q1<-10**Q[1:length(Q)] #Calculate the inverse of the log10 of the binned discharge
##

df3<-df2
for (j in 1:length(CL)) df3[,j]<-1-pnorm(df2[,j]) #Take the inverse of the standard normal z variate for each AEPz calcualted in step 5
rownames(df3)<-Q1
colnames(df3)<-CL


####################
# Step 8. Plot the data from the previous step:
x11()
pcol <- rep(rainbow(5),length(Q))
upperY<-ifelse(Extrapolate=="Yes",1.2,0.012)
plot(NA,NA,ylim=c(0,upperY),xlim=c(0,1), ylab="AEP", xlab="CL")
#for (i in 1:length(CL)) abline(v=CL[i], col="gray")

for (i in 1:length(Q)){
  lines(CL,df3[i,], type="l",col = pcol[i])
  points(CL,df3[i,], pch=20,col = pcol[i])
}



####################
# Step 9 & 10. Calculate the mean (expected) value of the AEP for each flow; this mean is equal to the area under the CDF. Then calulate the standard normal z variate for the mean AEP:

AEPm<-NULL #mean AEP
for (i in 1:length(Q)) AEPm<-c(AEPm,(trapz(x=CL,y=as.numeric(df3[i,])) + df3[i,1]*0.01 + df3[i,length(CL)-1]*.01)) #Use the trapezoidal rule for integration

AEPmz<-NULL
for (i in 1:(length(AEPm))) AEPmz<-c(AEPmz,qnorm((1-AEPm[i]))) #Calculate the standard normal z variate for each AEP:


####################
# Step 11. Plot the data from the previous step:
plot.new()
plot(NA,NA,ylim=c(2.5,6.5),xlim=c(-2.5,7.5), ylab="Log(Q)", xlab="AEP(z)")
lines(AEPz,df1[,"0"], type="l",col = "orange")
points(AEPz,df1[,"0"],pch=20,col = "orange")
lines(AEPmz,Q, type="l",col = "blue")
points(AEPmz,Q,pch=20,col = "blue")
legend("bottomright", c("Mean/Expected","Median/Computed"),col=c("blue","Orange"),lty=c(1,1))


####################
# Step 12. Calculate the standard normal z variate of a set of standard reporting values for AEP, calculate the mean log flow and the median log flow:

AEP1<-c(0.9, 0.5, 0.2, 0.1, 0.05, 0.02, 0.01, 0.005, 0.002, 0.001, 5E-04, 2E-04, 1E-04, 5E-05, 2E-05, 1E-05, 5E-06, 2E-06, 1E-06)
#Selected standard reporting AEP values

AEP1z<-NULL
for (i in 1:(length(AEP1))) AEP1z<-c(AEP1z,qnorm((1-AEP1[i]))) #Convert the AEP values to their standard normal z variate
    
f<-approxfun(AEPmz, Q) #Create a function for calculating the mean log flow for each AEP
Q2<-rep(0,length(AEP1z))

            
for (i in 1:(length(AEP1z))){
    if  (AEP1z[i]<min(AEPmz, na.rm=TRUE)){   #if the AEP standard normal variate is less than the minimum value used for the interpretation then:
        Q2[i]<-min(Q, na.rm=TRUE)
    } else {
        Q2[i]<-f(AEP1z[i])
    }
}
      
Q3<-rep(0,(length(Q))) #Create an empty array to store the inverse of the log10 of the discharge for the mean AEP
Q3<-10**Q2[1:length(Q2)]

f1<-approxfun(AEPz, df1[,"0"]) #Create a function for interpolating the median log flow for each AEP
Q4<-rep(0,length(AEP1z))

for (i in 1:(length(AEP1z))) Q4[i]<-f1(AEP1z[i])

Q5<-rep(0,length(Q4)) #Create an empty array to store the inverse of the log10 of the discharge for the median AEP
Q5<-10**Q4[1:length(Q4)]


####################
# Step 13. Summarize the final flow frequency curves:

df4<-data.frame(AEP=AEP1,Q_M_Haz_cfs=Q3,Q_Comp_Haz_cfs=Q5)
write.csv(df4,"sample_data/summary.csv")
print(df4)


####################
#Step 14. Plot the mean and computed flow frequency curves.
x11()
plot(NA,NA,ylim=c(100,100000),xlim=c(0.9,range(AEP1z,na.rm=TRUE)[2]), ylab="Flow(cfs)", xlab="AEP", log="xy", yaxt="n", xaxt="n")

lines(AEP1z, Q3, type="l", col="blue")
points(AEP1z, Q3, pch=20, col="blue")
lines(AEP1z, Q5, type="l", col="orange")
points(AEP1z, Q5, pch=20, col="orange")

aty <- c(1:5)
atx <- axTicks(1)
labelsy <- sapply(aty,function(i) as.expression(bquote(10^ .(i))))
labelsx <- atx
axis(2,at=10^aty,labels=labelsy)
axis(1,at=atx,labels=labelsx)

legend("bottomright", c("Mean/Expected","Median/Computed"),col=c("blue","Orange"),lty=c(1,1))

############
# END
############