from backtester.executionSystem.simple_execution_system import SimpleExecutionSystem
from backtester.logger import *
import numpy as np
import pandas as pd
from scipy.stats import norm


class Problem2ExecutionSystem(SimpleExecutionSystem):
    def __init__(self, enter_threshold=0.7, exit_threshold=0.55, longLimit=10,
                 shortLimit=10, capitalUsageLimit=0, enterlotSize=1, exitlotSize = 1, limitType='L', price='close', predictionType='b'):
        self.priceFeature = price
        self.predictionType = predictionType
        super(Problem2ExecutionSystem, self).__init__(enter_threshold=enter_threshold,
                                                                 exit_threshold=exit_threshold,
                                                                 longLimit=longLimit, shortLimit=shortLimit,
                                                                 capitalUsageLimit=capitalUsageLimit,
                                                                 enterlotSize=enterlotSize, exitlotSize=exitlotSize, limitType=limitType, price=price)

    def getPriceDf(self, instrumentsManager):
        instrumentLookbackData = instrumentsManager.getLookbackInstrumentFeatures()
        try:
            price = instrumentLookbackData.getFeatureDf(self.priceFeature)
            return price
        except KeyError:
                logError('You have specified Dollar Limit but Price Feature Key %s does not exist'%self.priceFeature)


    def exitPosition(self, time, instrumentsManager, currentPredictions, closeAllPositions=False):

        instrumentLookbackData = instrumentsManager.getLookbackInstrumentFeatures()
        positionData = instrumentLookbackData.getFeatureDf('position')
        position = positionData.iloc[-1]
        price = self.getPriceSeries(instrumentsManager)
        executions = pd.Series([0] * len(positionData.columns), index=positionData.columns)

        if closeAllPositions:
            executions = -position
            return executions
        executions[self.exitCondition(currentPredictions, instrumentsManager)] = -np.sign(position)*np.abs(position)
        executions[self.hackCondition(currentPredictions, instrumentsManager)] = -np.sign(position)*np.abs(position)
        # print('exit?',self.exitCondition(currentPredictions, instrumentsManager))
        return executions

    def enterPosition(self, time, instrumentsManager, currentPredictions, capital):
        instrumentLookbackData = instrumentsManager.getLookbackInstrumentFeatures()
        positionData = instrumentLookbackData.getFeatureDf('position')
        position = positionData.iloc[-1]
        price = self.getPriceSeries(instrumentsManager)
        # import pdb;pdb.set_trace()
        executions = pd.Series([0] * len(positionData.columns), index=positionData.columns)
        executions[self.enterCondition(currentPredictions, instrumentsManager)] = \
            self.getEnterLotSize(positionData.columns, price) * self.getBuySell(currentPredictions, instrumentsManager)
        # No executions if at position limit and we are adding to position
        executions[self.atPositionLimit(capital, positionData, price)&(self.getBuySell(currentPredictions, instrumentsManager)==position)] = 0
        # print('enter?', self.enterCondition(currentPredictions, instrumentsManager))
        # print(self.getBuySell(currentPredictions, instrumentsManager))
        return executions

    def getBuySell(self, currentPredictions, instrumentsManager):
        if self.predictionType=='b':
            return np.sign(currentPredictions - 0.5)
        else:
            return np.sign(currentPredictions)

    def enterCondition(self, currentPredictions, instrumentsManager):
        instrumentLookbackData = instrumentsManager.getLookbackInstrumentFeatures()
        if(len(instrumentLookbackData.getFeatureDf('prediction')) <= 2):
            return pd.Series(False, index=currentPredictions.index)
        else:
            if self.predictionType=='b':
                return (currentPredictions - 0.5).abs() > (self.enter_threshold - 0.5) #& ~((currentPredictions!=pastPredictions)&(currentPriceChange==pastPriceChange))
            else:
                return (np.abs(currentPredictions)>=norm.ppf(self.enter_threshold)*instrumentLookbackData.getFeatureDf('score').iloc[-1])

    def exitCondition(self, currentPredictions, instrumentsManager):
        instrumentLookbackData = instrumentsManager.getLookbackInstrumentFeatures()
        if(len(instrumentLookbackData.getFeatureDf('prediction')) <= 2):
            return pd.Series(False, index=currentPredictions.index)
        else:
            # printdf=pd.DataFrame(index=price.columns)
            # printdf['currentPredictions'] = currentPredictions
            # printdf['pastPredictions'] = pastPredictions 
            # printdf['currentPriceChange'] = currentPriceChange
            # printdf['pastPriceChange'] = pastPriceChange
            # print(printdf)
            if self.predictionType=='b':
                exit =  (currentPredictions - 0.5).abs() < (self.exit_threshold - 0.5) #| ((currentPredictions!=pastPredictions)&(currentPriceChange==pastPriceChange)) | ((currentPredictions==pastPredictions)&(currentPriceChange!=pastPriceChange))
            else:
                exit =  (np.abs(currentPredictions)<0.5*instrumentLookbackData.getFeatureDf('score').iloc[-1])

            ## Exit if thresholds are not satisfied or if we need to flip position
            return exit | (self.getBuySell(currentPredictions, instrumentsManager)!=instrumentLookbackData.getFeatureDf('position').iloc[-1])


    def hackCondition(self, currentPredictions, instrumentsManager):
        return pd.Series(False, index=currentPredictions.index)