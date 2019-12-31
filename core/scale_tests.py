from ras_ops import *

class ErrorClass(object):
    def __init__(self, error):
        self.Error = error

class ModelCheck(object):
    '''
    This object holds information for the files stored in a hec-ras zip file used for STARRII PFRA study.
    If path starts with s3 then the code will run from s3 zipfile, otherwise path is expected
    to be a string path to a zipped local model (e.g. *.zip)
    '''
    def __init__(self, model:RasModel):
        self.model = model
        self.contents = self.model.contents

    def junkFiles(self):
        check_data={}

        pList = [f for f in self.contents if pl.PurePosixPath(f).suffix[0:2]== '.p']
        pList = [ext for ext in pList if '.prj' not in ext]
        if pList.__len__() > 1: check_data['Plan'] =  pList

        gList = [f for f in self.contents if pl.PurePosixPath(f).suffix[0:2]== '.g']
        if gList.__len__() > 1: check_data['Geometry'] =  gList

        uList = [f for f in self.contents if pl.PurePosixPath(f).suffix[0:2]== '.u']
        if uList.__len__() > 1: check_data['Unsteady'] =  uList

        bList = [f for f in self.contents if pl.PurePosixPath(f).suffix[0:2]== '.b']
        bList = [ext for ext in bList if '.bc' not in ext and '.backup' not in ext and '.blf' not in ext]
        if bList.__len__() > 1: check_data['bFile'] =  bList

        xList = [f for f in self.contents if pl.PurePosixPath(f).suffix[0:2]== '.x' ]
        if xList.__len__() > 1: check_data['xFile'] =  xList

        cList = [f for f in self.contents if pl.PurePosixPath(f).suffix[0:2]== '.c' ]
        if cList.__len__() > 1: check_data['cFile'] =  cList

        dssList = [f for f in self.contents if pl.PurePosixPath(f).suffix[0:2]== '.dss' ]
        if dssList.__len__() > 1: check_data['dssFile'] =  dssList

        if check_data.items():
            return check_data
        else:
            return False

    def checkPrj(self, project:RasProject):
        check_data={}
        if project.plan_files.__len__() != 1: check_data['Plan'] = project.plan_files
        if project.geom_files.__len__() != 1: check_data['Geometry'] = project.geom_files
        if project.unsteady_files.__len__() != 1: check_data['Unsteady'] = project.unsteady_files

        if check_data.items():
            return check_data
        else:
            return False

    def checkPlan(self, planData:RasPlan):
        check_data={}
        if planData.geom_files.__len__() != 1: check_data['Geometry'] = planData.geom_files
        if planData.unsteady_files.__len__() != 1: check_data['Unsteady'] = planData.unsteady_files

        if check_data.items():
            return check_data
        else:
            return False

    def prereqFileChecks(self, localPlan: HDFPlanFile):
        check_data={}
        modelFiles = [str(pl.PurePosixPath(f).name) for f in self.contents]

        for f in localPlan.mandatoryFiles:
            if f not in modelFiles:
                check_data[f] = 'Simulation File Not Found'

        if check_data.items():
            return check_data
        else:
            return False

    def domainsCheck(self, localPlan: HDFPlanFile, forcing_data: RASForcing):
        check_data={}
        forcing_domains = list(forcing_data.domainForcing.keys())

        for fd in forcing_domains:
            if fd not in localPlan.domains:
                errorMessage = 'Domain {} in Forcing File Not Found in Model Domains List: {}'.format(fd, str(localPlan.domains))
                check_data[fd] = errorMessage

        if check_data.items():
            return check_data
        else:
            return False

    def s3Check(self):
        pass

    def timeSeriesCheck(self):
        # test tseries array length vs run days listed in file.
        pass

    def restartFileCheck(self):
        # Look in the b file, if restart = true, copy this file to the zip
        # Check to see if there is a date on the filename --> rename the file.
        pass

    def other(self):
        pass

    def runTests(self, model:RasModel, prjData:RasProject, planData:RasPlan, localPlan:HDFPlanFile, forcing_data: RASForcing):
        errors=[]

        fileTest = self.junkFiles()
        if fileTest:
            errors.append(fileTest)

        prjTest  = self.checkPrj(prjData)
        if prjTest:
            errors.append(prjTest)

        planTest = self.checkPlan(planData)
        if planTest:
            errors.append(planTest)

        preReqTest = self.prereqFileChecks(localPlan)
        if preReqTest:
            errors.append(preReqTest)

        domainCheck = self.domainsCheck(localPlan, forcing_data)
        if domainCheck:
            errors.append(domainCheck)

        return errors
