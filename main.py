from utils.annotate_table import annotate
from utils.clean_image import clean_image
from utils.extract_table import extract_table

if __name__ == "__main__":
    pdf_path = "./pdfs/Bank_Statement_Template_1_TemplateLab.pdf"
    
    outfile1 = "outfile1.png"
    annotate(pdf_path, out_filename= outfile1)
    
    outfile2 = "outfile2.png"
    clean_image(in_file=outfile1, out_file=outfile2)
    
    outfile3 = "outfile3.png"
    outfile4 = "outfile4_final.png"
    extract_table(outfile2, outfile3, outfile4)