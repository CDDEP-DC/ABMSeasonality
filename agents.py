import pandas as pd
import numpy as np
import itertools
import random
from collections import Counter
from datetime import datetime
import os
import sys
from scipy import signal

class Patient:
    def __init__(self, ID, hospital, unit, admissionDate, state, ambulatory, dischargeDate, contactPrecautions):
        self.ID = ID
        self.hospital = hospital
        self.unit = unit
        self.room = None
        self.admissionDate = admissionDate
        self.state = state
        self.ambulatory = ambulatory
        self.treatmentStartDay = None
        self.myNurse = None
        self.myDoctor = None
        self.dischargeDate = dischargeDate
        self.contactPrecautions = contactPrecautions
        self.contactCount = 0
        self.timer = 0
        self.log = []

    def __del__(self):
        del self
    
    def removePatient(self):
        # save history
        if self.hospital.monteCarlo == False:
            pd.DataFrame(self.log, columns=['day','hour','event']).to_csv(self.hospital.path+'/patients/patient_'+str(self.ID)+'.csv', index=False)
        # remove all instances
        self.room.patients.remove(self)
        # self.myNurse.patients.remove(self)
        # self.myDoctor.patients[self.unit.ID].remove(self)
        self.unit.patients.remove(self)
        self.hospital.patients.remove(self)

    def assignToHCW(self, day):
        # primary nurse
        myNurse = min(self.unit.nurses, key=lambda x: len(x.patients))
        myNurse.patients.append(self)
        self.myNurse = myNurse
        self.log.append([day, 0, 'assigned to nurse '+str(myNurse.ID)])
        # primary doctor
        myDoctor = min(self.hospital.doctors, key=lambda x: len(x.patients))
        myDoctor.patients[self.unit.ID].append(self)
        self.myDoctor = myDoctor
        self.log.append([day, 0, 'assigned to doctor '+str(myDoctor.ID)])

    def assignToRoom(self, day):
        emptyRooms = [room for room in self.unit.rooms if len(room.patients) == 0]
        myRoom = np.random.choice(emptyRooms)
        myRoom.patients.append(self)
        self.room = myRoom
        self.log.append([day, 0, 'assigned to room '+str(myRoom.ID)])
   
    def infectionTreatment(self, day, hour):
        if type(self.treatmentStartDay) == type(None):
            self.treatmentStartDay = day
            self.contactPrecautions = True
            self.log.append([day, hour, 'started AB treatment'])
            self.dischargeDate = max(self.dischargeDate, self.timer+self.hospital.transmissionParams['omega']+1)
        else:
            if day - self.treatmentStartDay > self.hospital.transmissionParams['omega']:
                self.treatmentStartDay = None
                self.contactPrecautions = False
                self.log.append([day, hour, 'ended AB treatment'])
                if random.random() < self.hospital.transmissionParams['rho']:
                    self.state = 'UC'
                else:
                    self.state = 'X'

    def doseResponseFunction(self, load):
        return load / (load + self.hospital.transmissionParams['E50'])
    
    def scheduleDailyVisits(self):
        nurse_contacts_min = self.unit.nurseContactsNCPmin
        nurse_contacts_max = self.unit.nurseContactsNCPmax
        doctor_contacts_min = self.unit.hospital.doctorContactsNCPmin
        doctor_contacts_max = self.unit.hospital.doctorContactsNCPmax
        if self.contactPrecautions:
            nurse_contacts_min = self.unit.nurseContactsCPmin
            nurse_contacts_max = self.unit.nurseContactsCPmax
            doctor_contacts_min = self.unit.hospital.doctorContactsCPmin
            doctor_contacts_max = self.unit.hospital.doctorContactsCPmax
        # nurse contacts
        nurse_contacts = np.random.randint(nurse_contacts_min, nurse_contacts_max)
        nurse_contacts_hour = nurse_contacts // 24 + 1
        nurse_hours = np.tile(np.arange(24), nurse_contacts_hour)
        random.shuffle(nurse_hours)
        nurse_hours = nurse_hours[:nurse_contacts]
        for h in nurse_hours:
            if random.random() < self.unit.primaryNurseVisitRate:
                nurse = self.myNurse
            else:
                nurse = np.random.choice(np.delete(self.unit.nurses, np.where(self.unit.nurses == self.myNurse)))
            nurse.visits[h].append(self)
        
        # doctor contacts
        doctor_contacts = np.random.randint(doctor_contacts_min, doctor_contacts_max)
        doctor_contacts_hour = doctor_contacts // 24 + 1
        doctor_hours = np.tile(np.arange(24), doctor_contacts_hour)
        random.shuffle(doctor_hours)
        doctor_hours = doctor_hours[:doctor_contacts]
        for h in doctor_hours:
            self.myDoctor.visits[h].append(self)


    def contactEnv(self, day, h):
        if self.hospital.transmissionPathways['patient_room']:
            # colonization
            if self.state in ['S','X'] and self.room.contamination == 1:
                p = self.hospital.transmissionParams['zeta']
                if self.state == 'X':
                    p *= self.hospital.transmissionParams['pi']
                if random.random() < p:
                    self.state = 'UC'
                    self.log.append([day, h, 'colonized from environment at room '+str(self.room.ID)])
                    self.unit.log.append([day, h, self.ID, 'colonized', 'environment at room '+str(self.room.ID)])
            elif self.state in ['UC','DC','I'] and self.room.contamination == 0:
            # shedding
                if random.random() < self.hospital.transmissionParams['eta']:
                    self.room.contamination = 1
        
        if self.hospital.transmissionPathways['patient_bathroom']:
            if (self.ambulatory == 1) and (random.random() < 1/6):
                b = np.random.randint(len(self.unit.bathrooms))
                # colonization
                if self.state in ['S','X']:
                    p = self.doseResponseFunction(self.unit.bathrooms[b].contamination)
                    if self.state == 'X':
                        p *= self.hospital.transmissionParams['pi']
                    if random.random() < p:
                        self.state = 'UC'
                        self.log.append([day, h, 'colonized from bathroom environment'])
                        self.unit.log.append([day, h, self.ID, 'colonized', 'environment at bathroom'+str(b.ID)])
                else:
                # shedding             
                    shedding = self.hospital.transmissionParams['eta_p']
                    if self.state in ['UC','DC']:
                        shedding *= self.hospital.transmissionParams['rho_C'] 
                    self.unit.bathrooms[b].contamination += shedding
                    

    def colonizationToInfection(self, day, h):
        if self.state in ['UC','DC']:
            if random.random() < self.hospital.transmissionParams['kappa']:
                self.state = 'I'
                self.log.append([day, h, 'infection symptom onset'])
                self.unit.log.append([day, h, self.ID, 'infected', 'colonization'])
                self.infectionTreatment(day, h)


    def regularTesting(self, day, h):
        if self.state == 'UC':
            if random.random() < self.unit.regularTesting * self.unit.testingAccuracy:
                self.state = 'DC'
                self.log.append([day, h, 'colonization detected'])
                self.contactPrecautions = True

    def decease(self, day, h):
        if random.random() < self.hospital.transmissionParams['delta']:
            self.log.append([day, h, 'deceased'])
            self.removePatient()
            self.__del__
    
    def discharge(self, day):
        self.log.append([day, 0, 'discharged'])
        self.removePatient()
        self.__del__()
        # terminal room disinfection
        if self.unit.terminalDisinfection == 1:
            self.room.disinfect()

        
class HCW:
    def __init__(self, ID, hospital, unit, contamination, hygieneComplianceEnter, hygieneComplianceExit, PPECompliance):
        self.ID = ID
        self.hospital = hospital
        self.unit = unit
        self.contamination = contamination
        self.PPEcontamination = 0
        self.contaminationHistory = contamination
        self.hygieneComplianceEnter = hygieneComplianceEnter
        self.hygieneComplianceExit = hygieneComplianceExit
        self.PPECompliance = PPECompliance
        self.PPE = False
        self.PPEcontamination = 0
        self.visits = [[] for i in range(24)]
        if self.unit != 'doctor':
            self.onDutyHours = np.random.randint(0, self.unit.nurseShift)
            self.patients = []
        else:
            self.patients = [[] for i in range(int(self.hospital.nUnits))]
        self.log = []

    def washHands(self, hygieneCompliance):
        # HCW hand hygiene
        if self.contamination == 1:
            if random.random() < hygieneCompliance * self.hospital.transmissionParams['sigmaHat_h']:
                self.contamination = 0
    
    def wearPPE(self):
        if random.random() < self.PPECompliance:
            self.PPE = True
    
    def doseResponseFunction(self, load):
        return load / (load + self.hospital.transmissionParams['E50'])
    
    def contactEnv(self, patient, day, h):
        if self.hospital.transmissionPathways['environmental']:
            # hcw contamination
            if patient.room.contamination == 1:
                if random.random() < self.hospital.transmissionParams['beta_eh']:
                    if not self.PPE:
                        self.contamination = 1
                        self.log.append([day, h, 'contaminated by environment at unit '+str(patient.unit.ID)+' room '+str(patient.room.ID)])
                    else:
                        self.PPEcontamination = 1
                        self.log.append([day, h, 'PPE contaminated by environment at unit '+str(patient.unit.ID)+' room '+str(patient.room.ID)])
            # shedding
            else:
                if (not self.PPE and self.contamination == 1) or (self.PPE and self.PPEcontamination == 1):
                    if random.random() < self.hospital.transmissionParams['beta_he']:
                        patient.room.contamination = 1

    def interactWithPatient(self, patient, day, h):
        if patient.contactPrecautions == 1:
            self.wearPPE()
        else:
            self.washHands(self.hygieneComplianceEnter)
        # env <-> hcw at entry
        self.contactEnv(patient, day, h)
        # patient to HCW
        if patient.state in ['UC','DC','I']:
            p = self.hospital.transmissionParams['beta_ph']
            if patient.state in ['UC','DC']:
                p *= self.hospital.transmissionParams['rho_C']
            if random.random() < p:
                if not self.PPE:
                    self.contamination = 1
                    self.log.append([day, h, 'contaminated by patient '+str(patient.ID)+' at unit '+str(patient.unit.ID)+' room '+str(patient.room.ID)])
                else:
                    self.PPEcontamination = 1
                    self.log.append([day, h, 'PPE contaminated by patient '+str(patient.ID)+' at unit '+str(patient.unit.ID)+' room '+str(patient.room.ID)])           
        # HCW to patient
        else:
            contamination = self.contamination
            if self.PPE:
                contamination = self.PPEcontamination
            if patient.state in ['S','X'] and contamination == 1:
                p = self.hospital.transmissionParams['beta_hp']
                if patient.state == 'X':
                    p *= self.hospital.transmissionParams['pi']
                if random.random() < p:
                    try:
                        dedicatedUnit = str(self.unit.ID)
                    except:
                        dedicatedUnit = 'doctor'
                    if random.random() < self.hospital.transmissionParams['epsilon']:
                        patient.state = 'I'
                        patient.log.append([day, h, 'infected by HCW: unit '+dedicatedUnit+', ID '+str(self.ID)])
                        patient.unit.log.append([day, h, patient.ID, 'infected', 'HCW unit '+dedicatedUnit+', ID '+str(self.ID)])
                        self.log.append([day, h, 'infected patient '+str(patient.ID)])
                        patient.infectionTreatment(day, h)
                    else:
                        patient.state = 'UC'
                        patient.log.append([day, h, 'colonized by HCW: unit '+dedicatedUnit+', ID '+str(self.ID)])
                        patient.unit.log.append([day, h, patient.ID, 'colonized', 'HCW: unit '+dedicatedUnit+', ID '+str(self.ID)])
                        self.log.append([day, h, 'colonized patient '+str(patient.ID)])
        # env <-> hcw at exit
        self.contactEnv(patient, day, h)
        if self.PPE:
            self.PPE = False
        else:
            self.washHands(self.hygieneComplianceExit)
            

    def contactStationBathroom(self, day, h):
        # shedding
        if self.contamination == 1:
            if self.hospital.transmissionPathways['nursing_station']:
                self.unit.station.contamination += self.hospital.transmissionParams['eta_h']
            if self.hospital.transmissionPathways['nurse_bathroom']:
                if random.random() < 1/6:
                    b = np.random.randint(len(self.unit.bathrooms))
                    self.unit.bathrooms[b].contamination += self.hospital.transmissionParams['eta_h']
        else:
            # contamination
            if self.hospital.transmissionPathways['nursing_station']:
                p = self.doseResponseFunction(self.unit.station.contamination)
                if random.random() < p:
                    self.contamination = 1
                    self.log.append([day, h, 'contaminated by environment at nursing station'])
            if self.hospital.transmissionPathways['nurse_bathroom']:
                if random.random() < 1/6:
                    b = np.random.randint(len(self.unit.bathrooms))
                    p = self.doseResponseFunction(self.unit.bathrooms[b].contamination)
                    if random.random() < p:
                        self.contamination = 1
                        self.log.append([day, h, 'contaminated by bathroom environment'])
                

    def changeShift(self, day, h):
        if abs(self.onDutyHours - self.unit.nurseShift) < 0.1:
            self.contamination = 0
            self.log.append([day, h, 'changed shift'])


class Room:
    def __init__(self, ID, unit, disinfectEfficacy, natClearRate, contamination):
        self.ID = ID
        self.unit = unit
        self.patients = []
        self.disinfectEfficacy = disinfectEfficacy
        self.natClearRate = natClearRate
        self.contamination = contamination
        self.contaminationHistory = []
    
    def disinfect(self):
        if random.random() < self.disinfectEfficacy:
            self.contamination = 0
    
    def naturalClearance(self):
        self.contamination *= (1 - self.natClearRate)
    


class Unit:
    def __init__(self, hospital, ID, name, rooms, capacity, bathrooms, patients, \
                 nursePatientRatio, bedUtilMin, bedUtilMax, dailyAdmissionMean, \
                 dailyAdmissionSD, admissionS, admissionX, admissionC, admissionI, \
                 LOSLogMean, LOSLogSD, disinfectionFrequency, terminalDisinfection, \
                 admissionTesting, regularTesting, regularTestingFreq, \
                 testingAccuracy, ambulatory, nurseShift, nurseHygieneComplianceEnter, \
                 nurseHygieneComplianceExit, nursePPECompliance, nurseContactsCPmax, \
                 nurseContactsCPmin, nurseContactsNCPmax, nurseContactsNCPmin, \
                 primaryNurseVisitRate):
        self.hospital = hospital
        self.ID = ID
        self.name = name
        self.rooms = rooms
        self.capacity = capacity
        self.bathrooms = bathrooms
        self.patients = patients
        self.nursePatientRatio = nursePatientRatio
        self.bedUtilMin = bedUtilMin
        self.bedUtilMax = bedUtilMax
        self.dailyAdmissionMean = dailyAdmissionMean
        self.dailyAdmissionSD = dailyAdmissionMean
        self.admissionStatus = [admissionS, admissionX, admissionC, admissionI]
        self.LOSLogMean = LOSLogMean
        self.LOSLogSD = LOSLogSD
        self.disinfectionFrequency = disinfectionFrequency
        self.terminalDisinfection = terminalDisinfection
        self.admissionTesting = admissionTesting
        self.regularTesting = regularTesting
        self.regularTestingFreq = regularTestingFreq
        self.testingAccuracy = testingAccuracy
        self.ambulatory = ambulatory
        self.nurseShift = nurseShift
        self.nurseHygieneComplianceEnter = nurseHygieneComplianceEnter
        self.nurseHygieneComplianceExit = nurseHygieneComplianceExit
        self.nursePPECompliance = nursePPECompliance
        self.nurseContactsCPmax = nurseContactsCPmax
        self.nurseContactsCPmin  = nurseContactsCPmin
        self.nurseContactsNCPmax = nurseContactsNCPmax
        self.nurseContactsNCPmin = nurseContactsNCPmin
        self.primaryNurseVisitRate = primaryNurseVisitRate
        self.station = None
        self.nurses = []
        self.dailyAdmissions = []
        self.dailyContacts = []
        self.stats = []
        self.log = []
        self.setup()

    def setup(self):
        # create patient rooms
        r = self.rooms
        self.rooms = []
        for i in range(r):
            self.rooms.append(Room(i, self, self.hospital.transmissionParams['sigmaHat_y'], self.hospital.transmissionParams['sigma_y'], 0))
        # create shared bathrooms
        b = self.bathrooms
        self.bathrooms = []
        for i in range(b):
            self.bathrooms.append(Room(i, self, self.hospital.transmissionParams['sigmaHat_w'], self.hospital.transmissionParams['sigma_w'], 0))
        # create nursing station
        self.station = Room(0, self, self.hospital.transmissionParams['sigmaHat_y'], self.hospital.transmissionParams['sigma_y'], 0)
        # create nurses
        n = int(np.round(self.nursePatientRatio * self.capacity))
        for i in range(n):
            self.nurses.append(HCW(i, self.hospital, self, 0, self.nurseHygieneComplianceEnter, self.nurseHygieneComplianceExit, self.nursePPECompliance))
        # create patients
        p = self.patients
        self.patients = []
        self.patients.extend(self.makeNewPatients(0, int(p)))

    
    def makeNewPatients(self, day, count, transfer=False):
        if transfer:
            source = 'external transfer'
        else:
            source = 'admission'
        newPatients = []
        admissionStatus = self.applySeasonality(day)
        count = min(count, self.capacity - len(self.patients))
        for i in range(count):
            if random.random() < self.ambulatory:
                amb = 1
            else:
                amb = 0
            ind = np.where(np.random.multinomial(1, admissionStatus))[0][0]
            state = ['S', 'X', 'C', 'I'][ind]
            CP = False
            if state == 'I':
                CP = True
            elif state == 'C':
                if np.random.random() < self.admissionTesting * self.testingAccuracy:
                    state = 'DC'
                    CP = True
                else:
                    state = 'UC'
                    
            los = max(1, np.round(np.random.lognormal(self.LOSLogMean, self.LOSLogSD)))
            newPatient = Patient(self.hospital.cumNumPatients, self.hospital, self, day, state, amb, int(los), CP)
            newPatient.log.append([day, 0, source+' to unit '+str(self.ID)])
            if state == 'UC':
                newPatient.log.append([day, 0, source+' as undetected colonized'])
                newPatient.unit.log.append([day, 0, newPatient.ID, 'colonized', source])
            elif state == 'DC':
                newPatient.log.append([day, 0, source+' as detected colonized']) 
                newPatient.unit.log.append([day, 0, newPatient.ID, 'colonized', source])
                newPatient.contactPrecautions = True
            elif state == 'I':
                newPatient.log.append([day, 0, source+' as infected']) 
                newPatient.unit.log.append([day, 0, newPatient.ID, 'infected', source])
                newPatient.infectionTreatment(day, 0)
            newPatient.assignToHCW(day)
            newPatient.assignToRoom(day)
            newPatients.append(newPatient)            
            self.hospital.cumNumPatients += 1
            self.hospital.patients.append(newPatient)
        return newPatients
    
    def applySeasonality(self, day):
        s0, x0, c0, i0 = self.admissionStatus
        c1 = c0 * self.hospital.importationSeasonality[day]
        s1 = ((1-c1-i0)*s0/x0) / (s0/x0+1)
        x1 = (1-c1-i0) / (s0/x0+1)
        return [s1, x1, c1, i0]

    def discharge(self, day):
        for patient in self.patients:
            if patient.state != 'I' and day >= patient.dischargeDate:
                patient.discharge(day)           

    def admission(self, day, transfer=False):
        n = np.round(np.random.uniform(self.bedUtilMin, self.bedUtilMax) * self.capacity) - len(self.patients)
        n = max(0, n)
        newPatients = self.makeNewPatients(day, int(n), transfer)
        self.patients.extend(newPatients)
        self.dailyAdmissions.append(int(n))                                    

    def transfer(self, day):
        n = sum((np.random.multinomial(1, list(self.dailyTransferProb.values())) > 0)*np.array(list(self.dailyTransferProb.keys())))
        newPatients = self.makeNewPatients(day, int(n), True)
        self.patients.extend(newPatients)
        
    def findPatient(self, patientID):
        for patient in self.patients:
            myPatient = None
            if patient.ID == patientID:
                myPatient = patient
                return myPatient
            if type(myPatient) == type(None):
                raise ValueError("Patient not found!")
      
    def writeStats(self):
        lst = Counter([p.state for p in self.patients])
        lst = [lst[i] for i in ['S','X','UC','DC','I']]
        lst2 = sum([n.contaminationHistory for n in self.nurses]) / 24
        lst3 = sum([d.contaminationHistory for d in self.hospital.doctors]) / 24
        self.stats.append([*lst, self.dailyAdmissions[-1], self.dailyContacts[-1], lst2, lst3])

        
class Hospital:
    def __init__(self, ID, name, nUnits, doctorPatientRatio, beds, doctorHygieneComplianceEnter, doctorHygieneComplianceExit, doctorPPECompliance, doctorContactsCPmax, doctorContactsCPmin, doctorContactsNCPmax, doctorContactsNCPmin, simLength, burnIn):
        self.ID = ID
        self.name = name
        self.nUnits = nUnits
        self.doctorPatientRatio = doctorPatientRatio
        self.beds = beds
        self.doctorHygieneComplianceEnter = doctorHygieneComplianceEnter
        self.doctorHygieneComplianceExit = doctorHygieneComplianceExit
        self.doctorPPECompliance = doctorPPECompliance
        self.doctorContactsCPmax = doctorContactsCPmax
        self.doctorContactsCPmin  = doctorContactsCPmin
        self.doctorContactsNCPmax = doctorContactsNCPmax
        self.doctorContactsNCPmin = doctorContactsNCPmin
        self.simLength = simLength
        self.burnIn = burnIn
        self.units = []
        self.doctors = []
        self.patients = []
        self.cumNumPatients = 0
        self.transmissionParams = {}
        self.monteCarlo = False
        self.fitToData = False
        self.seasonality = []

        self.setup()

    def setup(self):
        # transmission parameters
        params = pd.read_csv('./data/transmission_parameters.csv')
        paramsNames = params.loc[:,'parameters'].values
        paramsValues = params.loc[:,'value'].values
        for i,p in enumerate(paramsNames):
            self.transmissionParams[p] = paramsValues[i]
        # transmission pathways
        self.transmissionPathways = dict(np.array(pd.read_csv('./data/transmission_pathways.csv', header=None)))
        # seasonality pattern
        self.importationSeasonality = np.ones(self.simLength)
        if self.transmissionParams['seasonality_strength'] > 1:
            q = int(self.transmissionParams['seasonality_quarter'])
            t2 = np.linspace(-1, 1, 90, endpoint=False)
            e = signal.gausspulse(t2, fc=2.5, retquad=True, retenv=True)[2] * (self.transmissionParams['seasonality_strength'] - 1)
            self.importationSeasonality[((q-1)*90+self.burnIn):(q*90+self.burnIn)] += e 
        # create doctors
        d = int(np.round(self.doctorPatientRatio * self.beds))
        for i in range(d):
            self.doctors.append(HCW(i, self, 'doctor', 0, self.doctorHygieneComplianceEnter, self.doctorHygieneComplianceExit, self.doctorPPECompliance))  
        # create units
        unitsData = pd.read_csv("./data/units_parameters.csv")        
        for i in range(unitsData.shape[0]):
            self.units.append(Unit(self, *unitsData.iloc[i,:].values))              
        # create output folders
        self.path = './output/' + datetime.strftime(datetime.now(), '%Y_%m_%d_%H_%M')
        try:
            os.mkdir('./output/')
            os.mkdir(self.path)
            os.mkdir(self.path+'/patients')
            os.mkdir(self.path+'/units')
            os.mkdir(self.path+'/HCWs')
        except:
            pass
        
    
    def reset(self):
        self.units = []
        self.doctors = []
        self.patients = []
        self.cumNumPatients = 0
        self.transmissionParams = {}
        self.setup()
    
    def scheduleDailyVisits(self):
        for doctor in self.doctors:
            doctor.visits = [[] for i in range(24)]
        for unit in self.units:
            for nurse in unit.nurses:
                nurse.visits = [[] for i in range(24)]
            for patient in self.patients:
                patient.scheduleDailyVisits()

    def startDay(self, day):        
        for unit in self.units:
            for nurse in unit.nurses:
                nurse.contaminationHistory = nurse.contamination
            for patient in unit.patients:
                patient.timer += 1
                patient.contactCount = 0
                patient.colonizationToInfection(day, 0)
            unit.discharge(day)
            unit.admission(day)
            # natural clearance
            for room in unit.rooms:
                room.contaminationHistory.append(room.contamination)
                room.naturalClearance()
            unit.station.contaminationHistory.append(unit.station.contamination)
            unit.station.naturalClearance()
            for bathroom in unit.bathrooms:
                bathroom.contaminationHistory.append(bathroom.contamination)
                bathroom.naturalClearance()
            # schedule daily visits
            self.scheduleDailyVisits()
            ## environmental disinfection
            if unit.disinfectionFrequency > 0:
                if day % unit.disinfectionFrequency == 0:
                    for room in unit.rooms:
                        room.disinfect()
                    unit.station.disinfect()
                    for bathroom in unit.bathrooms:
                        bathroom.disinfect()
        for doctor in self.doctors:
            doctor.contaminationHistory = doctor.contamination

    def simulateDay(self, day):        
        # hourly events
        for h in range(24):
            for unit in self.units:
                for nurse in unit.nurses:
                    nurse.onDutyHours += 1
                    ## change shift
                    nurse.changeShift(day, h) 
                    ## hcw-patient visit
                    for patient in nurse.visits[h]:
                        nurse.interactWithPatient(patient, day, h)
                        patient.contactCount += 1
                    nurse.contaminationHistory += nurse.contamination
                for patient in unit.patients:
                    ## patient-env contact
                    patient.contactEnv(day, h)

            for doctor in self.doctors:
                for patient in doctor.visits[h]:
                    ## hcw-patient visit
                    doctor.interactWithPatient(patient, day, h)
                    patient.contactCount += 1
                doctor.contaminationHistory += doctor.contamination
        # daily events        
        for patient in self.patients:
            ## background transmission
            if random.random() < self.transmissionParams['alpha_b']:
                if random.random() < self.transmissionParams['epsilon']:
                    patient.state = 'I'
                    patient.log.append([day, 23, 'infected from background'])
                    patient.unit.log.append([day, 23, patient.ID, 'infected', 'background'])
                    patient.infectionTreatment(day, 23)
                else:
                    patient.state = 'UC'
                    patient.log.append([day, 23, 'colonized from background'])
                    patient.unit.log.append([day, 23, patient.ID, 'colonized', 'background'])
            ## regular testing
            if (patient.unit.regularTestingFreq > 0) and (day % patient.unit.regularTestingFreq == 0):
                patient.regularTesting(day, h)
            ## end infection treatment
            if patient.state == 'I':
                patient.infectionTreatment(day, 23)

        ## disinfection
        for unit in self.units:
            if (unit.disinfectionFrequency > 0) and (day % unit.disinfectionFrequency == 0):
                unit.station.disinfect()
                for bathroom in unit.bathrooms:
                    bathroom.disinfect()
                for room in unit.rooms:
                    room.disinfect()

    def endDay(self):
        for u in self.units:  
            dailyContacts = 0
            for patient in u.patients:
                dailyContacts += patient.contactCount
            u.dailyContacts.append(dailyContacts)
            u.writeStats()

    def simulate(self):
        if self.monteCarlo == False:
            try:
                os.mkdir(self.path)
                os.mkdir(self.path+'/patients')
                os.mkdir(self.path+'/units')
                os.mkdir(self.path+'/HCWs')
            except:
                pass
        for day in range(self.simLength):
            self.startDay(day)
            self.simulateDay(day)
            self.endDay()
        self.writeOutputs()
        
    def writeOutputs(self):
        if self.monteCarlo == False:
            for patient in self.patients:
                pd.DataFrame(patient.log, columns=['day','hour','event']).to_csv(self.path+'/patients/patient_'+str(patient.ID)+'.csv', index=False)
            for u in self.units:
                pd.DataFrame(u.stats, columns=['S','X','UC','DC','I','admissions','contacts','N1','D1']).to_csv(self.path+'/units/unit_'+str(u.ID)+'_stats.csv')
                pd.DataFrame(u.log, columns=['day','hour','patient_ID','event','source']).to_csv(self.path+'/units/unit_'+str(u.ID)+'_log.csv', index=False)
                load = []
                names = []
                for room in u.rooms:
                    load.append(room.contaminationHistory)
                    names.append('room_'+str(room.ID))
                for b in u.bathrooms:
                    load.append(b.contaminationHistory)
                    names.append('bathroom_'+str(b.ID))
                load.append(u.station.contaminationHistory)
                names.append('nursing_station')
                pd.DataFrame(np.transpose(load), columns=names).to_csv(self.path+'/units/unit_'+str(u.ID)+'_load.csv', index=False)
                for nurse in u.nurses:
                    pd.DataFrame(nurse.log, columns=['day','hour','event']).to_csv(self.path+'/HCWs/nurse_'+str(nurse.ID)+'_unit_'+str(nurse.unit.ID)+'.csv', index=False)
            for doctor in self.doctors:
                pd.DataFrame(doctor.log, columns=['day','hour','event']).to_csv(self.path+'/HCWs/doctor_'+str(doctor.ID)+'.csv', index=False)
    
     
       
        



