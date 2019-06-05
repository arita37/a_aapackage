# -*- coding: utf-8 -*-
"""
yahoo.db

daily_us_etf
daily_us_stock
daily_us_fundamental

ustef_all = np.array( ['SPY','AMJ','AAXJ','ACWI','ACWV','ACWX','AGG','AGZ'])

"""
%load_ext autoreload
%autoreload 2

import os
import sys

import sqlalchemy as sql

import data.hist_data_storage as yhh
import data.yahoo as yh
import util

os.chdir('D:\\_devs\\Python01\\project27\\')
sys.path.append(os.getcwd()+'/aapackage');  sys.path.append(os.getcwd()+'/linux/aapackage')
util.a_run_ipython("run " + os.getcwd() + '/aapackage/allmodule.py')
DIR= os.getcwd()


dbname= 'sqlite:///aaserialize/store/yahoo.db'
dstore= yhh.dailyDataStore(dbname)



dbtable= dstore.get_tablelist()





start1= util.date_now(-np.random.randint(2,25))
now1=    util.date_now(np.random.randint(2,45))   # now1='20160519'
print start1, now1

start1= "19900101"
now1="20160915"

#####################################################################################
############  Download Data from Yahoo ##############################################
#US ETF
ustef_all = np.array( ['SPY','AMJ','AAXJ','ACWI','ACWV','ACWX','AGG','AGZ','AMLP','AOR','ASHR','BAB','BBH','BIL','BIV','BKLN','BLV','BND','BNDS','BNDX','BNO','BOND','BSCH','BSCL','BSJH','BSJI','BSJJ','BWX','CGW','CHIE','CHII','CHIM','CHIQ','CHIX','CIU','CLY','CMBS','CMF','CORP','CRED','CSJ','CWB','DBA','DBC','DBEF','DBEU','DBGR','DBJP','DBO','DBS','DEM','DES','DEW','DFE','DFJ','DGL','DGRW','DHS','DIA','DJP','DLN','DON','DSI','DTN','DVY','DVYE','DWM','DWX','DXJ','ECH','EDV','EEM','EEMV','EFA','EFAV','EFG','EFV','EIDO','EMB','EMLC','EMLP','EPHE','EPI','EPP','EUFN','EWA','EWC','EWD','EWG','EWH','EWI','EWJ','EWL','EWM','EWN','EWP','EWQ','EWRE','EWS','EWT','EWU','EWW','EWX','EWY','EWZ','EZA','EZU','FBT','FCG','FDL','FDN','FENY','FEP','FEX','FEZ','FHLC','FLOT','FM','FNDA','FNDF','FNDX','FNX','FPE','FPX','FTA','FTC','FV','FVD','FXA','FXB','FXC','FXCH','FXD','FXE','FXF','FXG','FXH','FXI','FXL','FXO','FXU','FXY','GDX','GDXJ','GLD','GOVT','GSLC','GUNR','GVI','GXC','HACK','HDGE','HDV','HEDJ','HEFA','HEWG','HEWJ','HEZU','HYD','HYEM','HYG','HYLS','HYS','IAI','IAK','IAT','IAU','IBB','ICF','IDU','IDV','IEF','IEFA','IEI','IEMG','IEO','IEUR','IEV','IEZ','IFGL','IGE','IGF','IGM','IGOV','IGV','IHDG','IHE','IHF','IHI','IJH','IJJ','IJK','IJR','IJS','IJT','ILB','ILF','INC','INDA','INDY','IPS','IQDF','ITA','ITB','ITE','ITIP','ITM','ITOT','ITR','IUSG','IVE','IVV','IVW','IWB','IWC','IWD','IWF','IWM','IWN','IWO','IWP','IWR','IWS','IWV','IWY','IXC','IXJ','IXP','IXUS','IYC','IYE','IYF','IYG','IYH','IYJ','IYK','IYM','IYR','IYT','IYW','IYZ','JJA','JKD','JNK','JO','KBE','KIE','KRE','LQD','MBB','MCHI','MDY','MGC','MGK','MGV','MLPI','MOO','MTUM','MUB','NANR','NFO','NOBL','NTG','OEF','OIH','OIL','ONEQ','OUSA','PCY','PDP','PEJ','PEY','PFF','PGF','PGX','PHB','PHDG','PHYS','PID','PIN','PJP','PKW','PPLT','PRF','PSCD','PSCE','PSCF','PSCH','PSCI','PSCM','PSCT','PSCU','PSLV','PTLC','PWV','PXF','PXH','PZA','QAI','QDF','QLTA','QLTB','QLTC','QQQ','QUAL','RCD','REM','RFG','RGI','RHS','RPG','RPV','RSP','RSX','RTM','RWO','RWR','RWX','RYE','RYF','RYH','RYT','RYU','RZV','SCHA','SCHB','SCHC','SCHD','SCHE','SCHF','SCHG','SCHH','SCHM','SCHP','SCHV','SCHX','SCHZ','SCZ','SDIV','SDOG','SDY','SGOL','SHM','SHY','SHYG','SLV','SLYG','SMH','SOXX','SPHD','SPHQ','SPLV','SPYB','SRLN','STIP','STPZ','TFI','THD','TILT','TIP','TLH','TLO','TLT','TLTD','TLTE','TOTL','TUR','UGA','UNG','USDU','USMV','USO','UUP','VAW','VB','VBK','VBR','VCIT','VCLT','VCR','VDC','VDE','VEA','VEU','VFH','VGIT','VGK','VGLT','VGT','VHT','VIG','VIIX','VIS','VMBS','VNM','VNQ','VNQI','VO','VOE','VOO','VOOG','VOOV','VOT','VOX','VPL','VPU','VSS','VT','VTEB','VTI','VTV','VUG','VV','VWO','VWOB','VXF','VXUS','VXX','VXZ','VYM','WREI','XBI','XES','XHB','XLB','XLE','XLF','XLG','XLI','XLK','XLP','XLU','XLV','XLY','XME','XMLV','XOP','XPH','XRT','XSD','XSLV','XT','ZROZ' ])

dstore.download_todb(ustef_all, table1= 'daily_us_etf', start1=start1, end1=now1)




#US stocks
sp2000= np.array(['SNY','NKE','UPS','BMY','AGN','NTT','RY','NVO','QCOM','XRX','CHRW','HBI','CBG','IMS','MOS','LEN','BSAC','NTAP','TCK','EMN','SJR','WWAV','LPL','DRE','CNHI','UDR','ASX','LN','GOLD','GPS','ALLY','WYNN','DISCA','FBHS','TSO','RMD','ALV','CE','AKAM','HOG','RACE','PSO','ZNH','ALB','EXPD','SXL','CIB','GRMN','SEE','IPG','ARMK','Q','VAR','CQP','WFM','CDK','BPL','DOX','XYL','COO','CNA','TSS','JBHT','TRIP','SNPS','SNP','LFC','WBA','UTX','HON','LLY','BA','CELG','TD','SBUX','BHP','VOD','UNP','AJG','JNPR','FL','NFX','PVH','BG','LNT','N','ARE','SNA','ALK','TIF','JWN','ECA','RL','BRX','AES','LEA','PNW','AEG','KORS','REG','EEP','BR','MSCI','MNK','VAL','ANSS','VRSN','IHG','AOS','UNM','NCLH','MBT','TFX','COTY','VIPS','KSS','GT','Y','RE','WGP','RJF','UGI','SNI','AR','CMA','SPLK','RAD','NWSA','RDY','SABR','IVV','MTU','BBL','ACN','USB','CHTR','LMT','DEO','RAI','PCLN','AER','DRI','FFIV','DXCM','AMG','CCK','CPT','SPB','HBAN','MPEL','ACH','NI','WR','BWA','ATO','HII','WYN','TMK','TRGP','OTEX','FANG','PKG','KT','ALGN','JAZZ','NNN','MELI','CPL','IT','CDNS','DPZ','TKC','URI','ASH','NYCB','PAGP','SEIC','RPM','AIV','PHM','WES','FDS','CIT','CDW','OHI','ETFC','ZAYO','MIDD','MAA','WRB','FLR','AU','QRVO','FLEX','ALKS','SRCL','WPC','H','BBBY','STM','ITC','GIL','IEX','GLPI','RGA','SWN','VIP','LEG','SQM','LII','TXN','SPG','GS','BIIB','AVGO','COST','MDLZ','LOW','SAN','CL','BIDU','BNS','AIG','ITUB','AXP','ABT','TWX','HDB','DD','MS','BLK','DOW','RRC','PE','AVY','JKHY','WAB','CSGP','SGEN','DDR','AXTA','IEP','YNDX','YPF','WUBA','OAK','LUK','TRMB','EQGP','SSNC','HP','ALLE','UHAL','ENBL','KRC','CSL','ACC','ELS','EDU','MIC','DKS','SRC','TEAM','WLK','ST','AGNC','YZC','AFG','TSU','CSC','EQM','OGE','QGEN','MRVL','ROL','FTNT','SIG','SBNY','FTI','HDS','ENLK','FMC','OC','SBS','JEC','FLS','MKTX','HAR','ALNY','ZION','MD','STLD','FCE-A','LPT','TRU','ZG','WST','LAMR','ULTI','STE','TYL','BKFS','WBC','PF','ARW','TRQ','CB','TMO','AAPL','OXY','PBR','NEE','EMC','RIO','EPD','HMC','SHPG','E','DUK','UBS','GOOGL','DHR','CEO','GOOG','LYG','NGG','TEF','EOG','COP','ADBE','CRM','sto','BBD','TEVA','TJX','CNI','SO','F','BPY','CF','DISCK','PKI','HUBB','SPR','BURL','ODFL','MTN','VOYA','KAR','HLF','COMM','WOOF','DEI','PPC','SPLS','KGC','EGN','SNH','NDSN','PAC','AIZ','SIVB','CBOE','BAK','CPRT','GRA','ANET','BERY','FTR','SINA','WTR','STWD','PII','VEEV','POST','JLL','AMD','LYV','JBLU','SHLX','TARO','RS','SUI','ARRS','RGLD','TTC','AMH','UTHR','WRI','AVT','USFD','ERIE','EWBC','THS','HIW','OPK','GDDY','IM','BRO','SON','WFT','BEAV','KEYS','WSO','AXS','ARCC','PSXP','MIK','PACW','ABMD','NEU','TOL','SERV','GNTX','G','TFSL','KOF','PNRA','WCG','CG','CUBE','EPR','BUFF','SCI','PBCT','PTC','NFG','CHK','KMI','GM','AMT','ING','CAT','ABB','MET','D','MON','GD','PYPL','PUK','ESRX','KMB','FOXA','ASML','MSFT','PNC','SU','MFG','FDX','BK','LVS','SYK','BMO','ITW','REGN','NFLX','PSX','YHOO','MCK','BCE','SNE','RTN','ADP','TGT','AET','GIS','SYT','BBVA','SCHW','ORI','DNB','DLB','atr','AN','SMG','RNR','TLLP','RHI','BMS','XRS','DCI','HPT','MAN','ATHN','CRI','PNY','AM','CSAL','CLB','HRB','CBSH','IGT','LAZ','TPX','TCO','HTA','CQH','CASY','IAC','TGNA','MUR','MDU','HHC','ALSN','EPC','BMA','CPN','OZRK','FAF','BIO','GPK','UMC','ACM','NAVI','SPIL','ENDP','WSM','IPGP','AUY','STOR','QEP','PDCO','SIX','GWRE','AFSI','BAH','DNKN','ON','MOMO','ACHC','TSRO','ENH','MSCC','BOKF','NBIX','EQY','CNK','DCT','PINC','EV','MSM','WPX','HFC','WEX','SC','OA','CFR','STR','CAA','FEIC','GFI','DATA','MANH','MSG','CONE','LFL','THO','HTZ','CSRA','LSI','GGB','TER','QUNR','ASR','CNX','GXP','LECO','ACIA','GSH','HXL','CGNX','SNX','GEL','FLIR','UBNT','ICLR','APU','RICE','SBH','NCR','EVHC','BC','CBD','POOL','URBN','RPAI','GGG','NUAN','GPT','HEI','JBL','VVC','SNV','Z','CACC','FICO','HR','VR','NICE','BWP','TECH','EEFT','HPP','SQ','TDC','BPOP','TX','ARES','BWXT','BFAM','RSPP','HUN','PTHN','DST','ORAN','CAJ','ENB','PSA','AMX','JD','NOC','HPE','FB','AMZN','BDX','TMUS','HAL','YUM','BCS','XOM','COF','TRP','EBAY','CME','ECL','INFY','MMC','CTSH','CCL','CUK','BRK-B','PX','LILA','PRU','CNQ','KEP','APD','TRV','EMR','ICE','NVDA','LBTYA','NOK','cci','BSX','SPGI','BAM','ANTM','JNJ','STZ','AMAT','LYB','CI','AEP','EL','EXC','BX','TLK','ATVI','CM','PCG','PXD','TRI','VMW','FMX','BBT','ETN','TSLA','RBS','WPPGY','AFL','USG','PGRE','OMF','PAG','CW','FSLR','BBRY','MTCH','RAX','EQC','BVN','COLM','GWR','XPO','IDA','FBR','DPM','SKT','ACAD','AGCO','POR','HRC','NRG','PWR','WAL','SKX','COR','RLGY','CRL','CPHD','CY','TEGP','MMS','WWD','EXP','AUO','AZPN','AMCX','SYNT','PB','TDY','MPW','HAIN','NS','LOGI','CR','SID','IONS','CCJ','RBA','BFR','ISBC','GPOR','ICPT','VSAT','LANC','FCNCA','TRN','AGO','CIG','ALR','TTWO','PVTB','CLGX','CBRL','SATS','BRCD','ACAS','LM','CST','CFX','HLS','SBGL','AMSG','NRZ','BRKR','ZBRA','LILAK','RIG','R','RRD','GRUB','VWR','NATI','PPS','AWH','UMPQ','TCP','MORN','TRCO','FHN','ERJ','W','WBS','EME','EPAM','PBI','CAR','PRXL','FR','SWX','APO','RYN','TEP','RGC','SUN','AKRX','DLX','OLN','SHOP','LNCE','JCOM','CAB','P','NUVA','NUS','BSM','SXT','AEO','CPA','FII','THG','OGS','ELLI','EDR','JACK','FIT','BKD','DFT','TUP','APLE','FUN','HE','RES','X','PFPT','BLKB','HOMB','BKU','ITT','IART','CMN','XON','CRUS','SLM','PDM','HRG','INT','WGL','NYLD-A','NHI','USM','CIEN','TWO','CBT','GGAL','CLC','CIM','MDSO','TEN','CAVM','CMPR','SFR','TXRH','MOH','CBPO','JUNO','BGS','MPWR','BKH','HTHT','GRPN','PDCE','JW-A','ANAT','MRD','PSB','SM','ENLC','PAYC','RLI','LDOS','FLO','ENR','TDS','PRAH','JCP','AL','BOH','OUT','STRZA','KITE','ENS','ALE','CBI','YELP','CXP','MCY','PSEC','LGF','AMC','LPI','LHO','FDP','NXPI','KR','SYY','AON','DAL','APC','HCA','MNST','GE','SE','NTES','INTU','PLD','PPG','ALXN','WM','K','CRH','HCN','CHT','CS','CHU','PHG','NSC','STT','TSN','CSX','FMS','MFC','VALE','ORLY','BSBR','ISRG','EQIX','SRE','DE','HUM','SHW','VLO','IMO','T','WFC','FIS','LNKD','ALL','CAH','ZTS','GGP','VFC','ZBH','NWL','VTR','S','ADM','HPQ','BAX','BABA','CHL','JPM','ROST','ILMN','EA','EW','AVB','IBN','VIV','GLW','EIX','PG','EQR','DFS','WY','LUV','LUX','CP','VRTX','WPZ','PPL','HLT','ERIC','CBS','SYF','DISH','WMT','ED','FISV','DVN','WMB','MYL','TEL','MPC','STJ','TAP','NEM','PAYX','ABX','CERN','BXP','VZ','PFE','RCI','XEL','AZO','STI','LB','BHI','MCO','PEG','CTRP','HSY','NVS','RDS-A','TM','BUD','APA','BEN','ABC','ETP','OMC','LBTYK','AAL','IP','DG','FLT','PCAR','SIRI','CMI','HRL','PTEN','HZNP','OI','BWLD','RDN','KEX','DOC','SR','NJR','SLGN','AKR','CTLT','ASB','VMI','BDN','MUSA','MASI','NBR','EAT','UBSI','LSTR','UMBF','SAIC','MFA','BDC','CVG','PRA','SFM','GME','SAVE','STAY','PAAS','MEOH','VA','POL','SHO','LPX','ZEN','VGR','MBFI','RLJ','MANU','NWE','WNR','AHL','WTFC','VLP','BXMT','UNVR','LFUS','NVRO','WOR','PZZA','OSK','IBKC','CNO','OFC','OLED','MTG','YY','HCSG','PRI','CHH','IDTI','CLH','ESRT','RARE','SUM','RBC','SBGI','EXEL','JNS','SLCA','JOY','NTCT','RHP','EGO','WEN','NKTR','MDCO','QTS','TEX','NGD','ISIL','ATHM','LPLA','FEYE','BECN','AVA','CCP','INXN','TMH','RRR','TKR','FNB','DY','MENT','TSE','SWFT','HA','GEF','UNF','MKSI','PNM','TECD','PNFP','AVP','PFGC','COHR','FNSR','LXP','PBH','GWPH','CLI','PSMT','WPG','WCC','CMP','PODD','WMGI','HBHC','SF','CSOD','HELE','SPN','KATE','CREE','KW','BGCP','NRF','LPNT','INOV','IDCC','CHDN','OII','TCB','HNI','FIVE','CC','CAKE','MDRX','ROIC','ESNT','FULT','VC','CACI','VLY','THC','IOC','SIR','KRG','DW','FUL','AWI','CATY','EGP','TR','CBL','KOS','NSAM','ENTG','CCO','HL','ORA','EVER','GNRC','WAFD','TCBI','FFIN','ALGT','ZNGA','LGND','VSTO','WRE','NGHC','MTX','FSIC','RDUS','PSTG','MTZ','MFRM','INCR','UFS','SLAB','GNW','DAR','CUZ','AMBA','CDE','FLTX','SJI','SIGI','ELP','CVLT','CNV','VIRT','KNX','CRTO','WWW','BXS','AVX','PCTY','KYN','MTSI','WAGE','FCS','STL','MDP','MEG','ESL','JJSF','LXK','PBF','GMED','CRZO','NORD','MFS','KMT','ACIW','PAY','BYD','SAM','LOGM','GBCI','DO','RXN','FIZZ','HTH','WTS','BLMN','EFII','NEOG','ISCA','BCPC','CHE','DNOW','PEN','EVR','DORM','WLL','ESV','VRNT','DAN','AMKR','MTDR','BID','HSNI','LITE','TPH','CALM','B','UNFI','BIG','IILG','UFPI','OMI','SSD','ABM','SAFM','LAD','MLHR','PGND','TMHC','MSA','SFUN','SHOO','COT','CLNY','DPLO','PEGI','KBR','LTC','DDS','MLNX','NYT','AB','PLNT','CBU','LC','PEB','PAH','VSH','AXE','DRQ','ITRI','TTEK','VAC','DOOR','ODP','IRWD','MOG-A','JBT','AAT','SSTK','RP','PKY','DBD','DSW','WBMD','SPH','HI','ACXM','ICUI','ASGN','CATM','ARIA','EXAS','SYNA','DECK','PRTY','PEGA','HUBS','V','PTR','CVX','SCCO','CAG','TU','DB','FOX','VNO','RYAAY','DLTR','TYC','APH','SLF','ETE','WEC','DLPH','NLSN','CCE','KO','PKX','MT','EC','ADI','BF-B','HCP','PGR','CXO','SWK','CLR','MTB','ROP','FTV','MRK','INTC','AZN','MAR','LVLT','UAL','MU','NMR','WIT','WLTW','RSG','O','ES','IR','CPB','WDAY','DPS','EXPE','TROW','UA','ORCL','HD','BAC','AMP','DTE','AMTD','CLX','TI','SSL','PH','BMRN','VIAB','MMP','SJM','HES','BCR','EFX','TS','SKM','NUE','CSCO','CMCSA','PM','PEP','IBM','DIS','LVNTA','MHK','QVCA','HIG','MGA','INFO','NTRS','TDG','INCY','TV','CTL','ULTA','A','BAP','ADSK','FITB','SYMC','ESS','LRCX','ROK','NBL','VMC','HSBC','TSM','RCL','DLR','WDC','MJN','GPC','SNN','POT','SBAC','HNP','IBKR','LH','ETR','MGM','CA','XRAY','FE','KB','PFG','VRSK','TWTR','C','UL','UN','MO','LLTC','BLL','FCX','L','GG','SEP','BRFS','WHR','GWW','LMCA','CTXS','XLNX','SWKS','AWK','WCN','MDVN','AA','MSI','GIB','CHKP','HSIC','MRO','KEY','RHT','AGU','DVA','IVZ','MKL','MCHP','HOT','PRGO','BSMX','AMGN','UNH','TOT','AGR','CMG','WAT','PANW','HST','KIM','FDC','XEC','MKC','CINF','NOV','CFG','ADS','CHD','UGP','RF','EQT','BBY','WRK','SLW','AEM','NOW','FAST','CTAS','CVE','AEE','KKR','MDT','BTI','MAC','AAP','CMS','DHI','AYI','SLG','GPN','DGX','COG','BIP','MPLX','KMX','NDAQ','UHS','MAS','HRS','LLL','M','PNR','CNC','MLM','FRC','CX','FRT','PAA','OKS','FNF','MXIM','QSR','AME','MAT','TSCO','KLAC','LNC','DOV','LBRDK','LKQ','IFF','TXT','SLB','SAP','KHC','MMM','ma','BP','GSK','ABBV','GILD','DCM','MCD','CVS','COL','MTD','STX','COH','MBLY','NLY','HOLX','WU','KSU','OKE','LULU','VRX','IRM','WB','LNG','XL','VNTV','EXR','VER','HAS','IDXX','ACGL','SCG','INGR','CNP','GPRO','CPE','DRH','IMAX','OPHT','DM','MLI','NVAX','CVA','SANM','FNGN','BLUE','MWA','APAM','CFFN','JOBS','ALEX','IBOC','MGEE','CZZ','CNS','LOPE','MATW','GLNG','ATI','HQY','EEQ','MSTR','KMPR','PBYI','FIG','HMHC','GVA','HAE','WSTC','CTB','ABY','FGP','KLXI','SNCR','APFH','ITCI','GWB','TRMK','HEP','GK','PRTA','CHMT','REV','SIMO','NGL','ONB','COLB','EBIX','CXW','CVBF','HAWK','EE','MNRO','NTGR','ROLL','PLAY','CEB','HMSY','PAM','AIT','OAS','CYBR','CMC','CMCM','FCN','SCTY','NG','RMP','FI','TERP','NEWR','ROVI','VIAV','BOX','SSB','AGIO','IVR','XHR','TREX','TEO','IPXL','MAIN','AAN','AEIS','IAG','TLN','STAG','CPS','ONCE','KS','WIX','BCO','FELE','MORE','TVPT','SMTC','CRS','MATX','SCS','PEI','ABCO','EXLS','AHS','EGBN','HIMX','SEMG','LXFT','SFLY','SCAI','AYR','HYH','CORE','AGII','WERN','GOV','AZZ','CHFC','POWI','PLCM','SBRA','DGI','ENV','WDFC','PLT','NWN','SWHC','BRC','CACQ','STMP','IPHI','TWOU','DDD','ALDR','FFG','RDC','ERF','FET','TGI','AMED','TSRA','CHS','CCOI','ETSY','NXST','HTLD','AXON','OMAM','HLI','SWC','RNG','BITA','SOHU','KALU','NNI','HMY','GIMO','WWE','SCL','DYN','OLLI','GEO','NYRT','VLRS','SMLP','LTRPA','FMBI','NWBI','FBC','AEL','AF','PLCE','SEM','FCB','PCH','FDML','OIS','PCRX','WNS','LXRX','NSM','PLXS','NAV','MANT','SSRI','ARLP','RPT','CWT','ZLTQ','AHGP','ASNA','BLDR','WDR','PRAA','SKYW','NOAH','AAON','SFNC','DF','GSAT','TIME','SSW','RMBS','LOCK','CHSP','TNET','EGOV','BANR','REXR','FN','IMPV','DIN','EDE','RNST','TOWN','AWR','NXTM','CEQP','HUBG','UCBI','GNC','GIII','LADR','MTH','PRK','EVH','SVU','OMCL','NSP','LTXB','CPPL','CSGS','SFL','DV','PRLB','JOE','PRGS','MYGN','GNL','NSR','MRC','SHLD','GCI','CBM','AMWD','PFS','IOSP','FWRD','NE','RH','GSM','KFY','TBPH','NBTB','NAVG','FGL','SYRG','FMSA','HALO','FPRX','LQ','SSP','SCHL','INDB','SPWR','ATU','TASR','FOSL','AINV','BABY','BSFT','CVT','APOG','HURN','VNOM','TPRE','FIBK','WETF','TXMD','WAIR','IIVI','AIN','SFBS','CLF','CYS','SHAK','UVV','NP','OTTR','GLOB','TTEC','SRPT','YDKN','BEL','QUAD','ORBK','KWR','FFBC','MINI','MC','SAGE','GES','BOFI','TSEM','CSTE','ININ','MGLN','OB','CBF','KBH','AGRO','AIRM','DEPO','BLD','ASTE','MDC','GPI','LZB','MAG','CLS','AXDX','HOLI','SYKE','AAV','EXPO','WMS','NEP','PZE','FSP','KNL','HW','CHRS','SONC','PTLA','AXL','BLOX','QLYS','CYOU','CUB','MEI','KANG','WABC','CYH','MDR','VG','INVA','AVG','EVTC','OSIS','WNRL','PDS','INFN','CVI','BETR','AJRD','KRNY','GMLP','CUDA','TRNO','THRM','BNFT','CENT','SHEN','ABCB','TOUR','MXL','LCI','ALRM','EGHT','KCG','FLOW','OCLR','KN','INN','OXM','CNSL','ROCK','SCSS','CYNO','SWM','KAMN','CCMP','ABG','OPB','SDRL','ANF','CEMP','PENN','CVGW','BKE','BNCN','INGN','BHE','DNR','FRO','SCOR','FGEN','RGR','NPO','HIFR','MCRN','ESE','ACHN','SPSC','XLRN','CNMD','SPTN','PLKI','SEAS','BNCL','EIGI','DERM','TREE','USAC','CALD','TILE','AMSF','TNC','NYLD','ARI','LABL','TPC','ADC','EGL','ACOR','IBP','FOE','GDOT','BLX','QTWO','ABAX','ATRO','ALOG','SUPN','ACCO','RGEN','WSFS','CAFD','TIVO','SSYS','CNNX','UFCS','GLOG','ELY','INSY','KRA','QUOT','EBS','FRME','RWT','NSIT','SXI','TMP','TTMI','GOGO','BANC','MMSI','IRBT','TSLX','UEIC','NCS','MBI','CAL','NSH','STC','EPE','ARCO','SMCI','FIX','MPG','AKS','MUX','DK','SPNC','MGNX','ATNI','CLW','TRTN','BPFH','PMT','HASI','OEC','ELGX','BCC','IPAR','TROX','TGP','MHLD','GCO','PAHC','SMP','HTGC','HF','CKH','ROG','ANDE','SAFT','CVRR','FINL','SNR','PRIM','FBP','HRI','GKOS','DIOD','GLT','EIG'])

dstore.download_todb(sp2000, table1= 'daily_us_stock2', start1=start1, end1=now1)






############ Risk Table ##############################################################







############  Get Data ##############################################################
#all in one dataframe
df= dstore.get_histo(['AAPL'], table1='daily_us_stock2', split_df=0)
  

#all in onesplitted  list
qlist= dstore.gethisto(ustef_all, table1='daily_us_etf', split_df=1)



df2= dstore.get_histo(['SPY'], table1='daily_us_etf', start1="20150101", end1="20170101",  split_df=0)



#Get list of Symbols
symlist= dstore.get_symlist(table1='daily_us_stock')



'daily_us_longhisto'










######################################################################################
#-------------Update Table Column-----------------------------------------------------
df= dstore.get_histo([], table1='daily_us_stock', start1="", end1="",  split_df=0)
df['date'] = df['date'].map(lambda x: int(x.replace('-', '')))   

util.sql_delete_table('daily_us_stock', 'sqlite://' + dbname)

#Batch
nchunk= 500000
for j in xrange(0,3):
  i=nchunk*j
  df2= df[i:i+nchunk]
  df2.to_sql('daily_us_stock', dstore.con, if_exists='append', index=False, index_label=None, chunksize=None)
######################################################################################











        


#####################################################################################
#Clearn DB Duplicate
dstore.cleandb( 'daily_us_etf' )
dstore.cleandb( 'daily_us_stock' )







util.sql_delete_table('daily', dbname)




conn = sqlite3.connect(dbname)
del conn

dbengine = sql.create_engine('sqlite://'+dbname) 
'''
https://addons.mozilla.org/en-US/firefox/addon/sqlite-manager/?src

http://sebastianraschka.com/Articles/2014_sqlite_in_python_tutorial.html
'''











yh.YFinanceDataExtr







############################################################################
#---------------------             --------------------
